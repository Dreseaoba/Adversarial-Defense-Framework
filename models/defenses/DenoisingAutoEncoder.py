import time
import os

import logging
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.modules.utils import _pair, _quadruple
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
from models.defenses.PIN.vision_image_folder import ImageFolder

from models.defenses.dataset import AdvDataset

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shortcut=True):
        super(EncoderBlock, self).__init__()
        self.shortcut = shortcut
        if shortcut:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                          #nn.BatchNorm2d(out_channels)
                                          )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1 ,bias=False)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.nolinear1 = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1 ,bias=False)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.nolinear2 = nn.PReLU(out_channels)

        
    def forward(self, x):
        if self.shortcut:
            shortcut_x = self.shortcut_layer(x)
        #x = self.nolinear1(self.bn1(self.conv1(x)))
        #x = self.nolinear2(self.bn2(self.conv2(x)))
        x = self.nolinear1(self.conv1(x))
        x = self.nolinear2(self.conv2(x))
        if self.shortcut:
            x = x + shortcut_x
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output=False):
        super(DecoderBlock, self).__init__()        
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.nolinear1 = nn.PReLU(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1 ,bias=False)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        if output:
            self.nolinear2 = nn.Tanh()
        else:
            self.nolinear2 = nn.PReLU(out_channels)

        
    def forward(self, x):
        #pdb.set_trace()
        #x = self.nolinear1(self.bn1(self.conv1(x)))
        #x = self.nolinear2(self.bn2(self.conv2(x)))
        x = self.nolinear1(self.conv1(x))
        x = self.nolinear2(self.conv2(x))
        return x

class DAE(nn.Module):
    def __init__(self, shortcut=True):
        super(DAE, self).__init__()
        # shortcut = True
        self.encoder1 = nn.Sequential(EncoderBlock(3, 64, stride=2, shortcut=shortcut),
                                     EncoderBlock(64, 128, stride=2, shortcut=shortcut),
                                     EncoderBlock(128, 256, stride=2, shortcut=shortcut))
        self.decoder1 = nn.Sequential(DecoderBlock(256, 128),
                                     DecoderBlock(128, 64),
                                     DecoderBlock(64, 3, output=True))
        self.encoder2 = nn.Sequential(EncoderBlock(3, 64, stride=2, shortcut=shortcut),
                                     EncoderBlock(64, 128, stride=2, shortcut=shortcut),
                                     EncoderBlock(128, 256, stride=2, shortcut=shortcut))
        self.decoder2 = nn.Sequential(DecoderBlock(256, 128),
                                     DecoderBlock(128, 64),
                                     DecoderBlock(64, 3, output=True))
        
    def forward(self, x):
        #pdb.set_trace()
        x1 = self.encoder1(x)
        x1 = self.decoder1(x1)
        x2 = self.encoder1(x1)
        x2 = self.decoder1(x2)

        # return x1
        return x2



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits       
    
class UnNormalize:
            #restore from T.Normalize
            #反归一化
            def __init__(self,mean=(0.485, 0.456, 0.406),std= (0.229, 0.224, 0.225),device=None):
                self.mean=torch.tensor(mean).view((1,-1,1,1)).to(device)
                self.std=torch.tensor(std).view((1,-1,1,1)).to(device)
            def __call__(self,x):
                x=(x*self.std)+self.mean
                return torch.clip(x,0,None)

class Loss():
    def __init__(self, norm_lambda=0.01):
        self.bce = nn.MSELoss()
        self.norm_lambda =norm_lambda

    def __call__(self, input:torch.Tensor, target):
        bceloss = self.bce(input, target)
        norm = self.norm_lambda * input.abs().mean()
        return bceloss + norm, bceloss, norm

class DenoisingAutoEncoder():
    
    def __init__(self, device=None, model='dae', res_output=False, ckpt=None) -> None:
        """
        """
        if model=='dae':
            self.model = DAE()
        elif model=='unet':
            self.model = UNet(3)
        self.transform = transforms.Compose([
                transforms.Resize(size = [384,384]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.transform_eval = transforms.Compose([
                transforms.Resize(size = [384,384]),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.device = device if device is not None else torch.device('cuda')
        # self.unnormalize = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], self.device)
        self.model = self.model.to(self.device)
        self.res_output = res_output

        self.curr_epoch = 0
        if ckpt is not None:
            checkpoint = torch.load(ckpt, map_location=lambda storage, loc:storage)
            self.model.load_state_dict(checkpoint['model'])
            self.curr_epoch = checkpoint['epoches']
        
    def noise_fn(self, inputs: torch.Tensor, scale: float, resize: float = 2):
        """
        add Gaussion noise to inputs
        sacle is the Sigmma of Gaission distribution
        """
        n,c,h,w = inputs.size()
        noise = torch.randn((n,c,h//resize,w//resize))
        noise = transforms.Resize(inputs.size()[-2:], interpolation=transforms.InterpolationMode.NEAREST)(noise)
        inputs_scale = ((inputs**2).sum())**0.5
        noise_scale = ((noise**2).sum())**0.5
        noise = noise*scale*inputs_scale / noise_scale
        noise = Variable(noise, requires_grad=False)
        inputs_noise = inputs + noise
        return inputs_noise

    def train(self,
        logger:logging.Logger,
        save_path:str, 
        max_epoch=20,
        batch_size=16,
        lr=0.01,
        sch='step',
        momentum=0,
        decay_step=2,
        decay_rate=0.5,
        noise_scale=0.04,
        norm_lambda=0.01,
        # save_step=2,
        data_path='/data/guohaoxun/workspace/atk/AdvDefenseFramework/datasets/deepfake/train_pca',
        noise_path=None,
        valid_path=None
        ):
        """DAE training

        Args:
            `save_path` (str): path to save checkpoint files
            `ckpt` (str): give when continue training, path of checkpoint to load
            `max_epoch` (int): max num of epochs to train. if continue from a checkpoint, the epoch num already run would be read from the ckpt file
            `momentum`, `decay_step`, `decay_rate`: optimizer params
            `noise_scale` (float, 0~1): the scale of Gaussion noise to be added to input data during training 
            `save_step` (int): the frequncy of saving ckpts
            `data_path` (str): path of dataset, should be a folder containing iamges 
        """
        # get data
        noise_training = noise_path is None
        valid = valid_path is not None

        print('LOADING DATA')
        if noise_training:
            data_set = ImageFolder(data_path, transform=self.transform)
        else:
            data_set = AdvDataset(data_path, noise_path, transform=self.transform)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
        if valid:
            valid_set = AdvDataset(os.path.join(valid_path, 'ori'), os.path.join(valid_path, 'adv'), transform=self.transform)
            valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        print(len(data_loader))

        # loss
        loss_fn = Loss(norm_lambda)

        # train
        optimizer = optim.SGD(params=self.model.parameters(), lr=lr, momentum=momentum)
        if sch == 'step':
            lr_sch = optim.lr_scheduler.StepLR(optimizer, decay_step, decay_rate, last_epoch=self.curr_epoch-1)
        elif sch == 'cos':
            lr_sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3, 2, last_epoch=self.curr_epoch-1)
        else:
            raise NotImplementedError('no scheduler named {}'.format(sch))
        self.model.train()

        print('START TRAINING')
        last_epoch_time = 0
        
        for epoch in range(self.curr_epoch, max_epoch):
            logger.info('====== epoch {}/{} ======'.format(epoch+1, max_epoch))
            epoch_time = time.time()
            running_loss = 0.
            epoch_loss = []
            for i, data in enumerate(data_loader, 0):
                # input data
                if noise_training:
                    inputs, _, _ = data
                    inputs = Variable(inputs)
                    inputs_noise = self.noise_fn(inputs, noise_scale)
                else:
                    inputs_noise, inputs = data
                inputs = inputs.to(self.device)
                inputs_noise = inputs_noise.to(self.device)

                # zero the grad
                optimizer.zero_grad()

                # forward
                result = self.model(inputs_noise)


                # loss and backward
                if self.res_output:
                    loss, _, norm = loss_fn(result, inputs - inputs_noise)
                else:
                    loss, _, norm = loss_fn(result, inputs)
                loss = loss.to(self.device)
                loss.backward()
                # loss.backward()
                optimizer.step()

                # print log
                running_loss += float(loss.item())
                # running_loss += float(loss.item())
                if (i+1)%100==0:
                    logger.info('epoch: {}, step: {}, loss: {}'.format(epoch+1, i+1, running_loss))
                    epoch_loss.append(running_loss)
                    running_loss = 0.
            epoch_time = time.time() - epoch_time
            if last_epoch_time==0:
                ETA = (max_epoch - epoch - 1) * epoch_time
            else:
                ETA = (max_epoch - epoch - 1) * (epoch_time + last_epoch_time)/2
            logger.info('====== epoch {} time: {:.2f} ETA: {:.2f} loss:{:.2f}======'.format(epoch+1, epoch_time, ETA, sum(epoch_loss)))
            lr_sch.step(epoch)

            # save model
            if epoch%decay_step==decay_step-1:
                if valid:
                    valid_loss = []
                    for i, data in enumerate(valid_loader, 0):
                        if noise_training:
                            inputs, _, _ = data
                            inputs = Variable(inputs)
                            inputs_noise = self.noise_fn(inputs, noise_scale)
                        else:
                            inputs_noise, inputs = data
                        inputs = inputs.to(self.device)
                        inputs_noise = inputs_noise.to(self.device)
                        optimizer.zero_grad()
                        result = self.model(inputs_noise)
                        if self.res_output:
                            loss, _, norm = loss_fn(result, inputs - inputs_noise)
                        else:
                            loss, _, norm = loss_fn(result, inputs)
                        loss = loss.to(self.device)
                        running_loss += float(loss.item())
                        if (i+1)%100==0:
                            valid_loss.append(running_loss)
                            running_loss = 0.
                    logger.info('====== epoch {} valid loss:{:.2f}======'.format(epoch+1, sum(valid_loss)/len(valid_loss)))
                ckpt_data = dict(
                        optimizer = optimizer.state_dict(),
                        model = self.model.state_dict(),
                        epoches = epoch+1,
                        )
                ckpt_name = os.path.join(save_path, 'ae'+'_'+str(epoch+1)+'.pth')
                torch.save(ckpt_data, ckpt_name)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.model.eval()
            x = x.to(self.device)
            x = self.transform_eval(x)
            output = self.model(x)
            if self.res_output:
                output += x
        return output