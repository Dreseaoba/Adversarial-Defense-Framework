import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange, reduce, repeat
import math
import numbers
import numpy as np
import cv2
import copy
from torch.nn.modules.utils import _pair, _quadruple
from torch.autograd import Variable
import logging
from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
from models.defenses.PIN.vision_image_folder import ImageFolder

from models.defenses.dataset import AdvDataset
import time
import torch.optim as optim
import os

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads # 64 x 8
        self.heads = heads # 8
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        print('1')
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        # 解释x=10,5,512的情形
        # pdb.set_trace()
        b, n, _, h = *x.shape, self.heads
        # b,n,_,h  ==  10,5,512,8
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # self.to_qkv(x) == 10,5,1536
        # self.to_qkv(x).chunk(3, dim = -1) == ((10,5,512),(10,5,512),(10,5,512)) 得到一个包含三个元素的tensor
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # map是对元组中的每个tensor都执行相同的操作
        # rearrange:把(10,5,512)看作(10,5,8*64)然后变形为(10,8,5,64)
        # q,k,v都是(10,8,5,64)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # 'bhid,bhjd->bhij'是说,两个矩阵求相似度,先把bhjd变为bhdj,然后bhid*bhdj->bhij
        # 就得到了bhij,形状是10,8,5,5
        # 理解为每个q和其它的k的相似度,也就是每个q得到5个值
        # 最后scale一下
        # dots:10,8,5,5
        # 不考虑mask这一块的内容
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        # pdb.set_trace()
        attn = dots.softmax(dim=-1)
        # print("attention weight",torch.sum(attn,dim=1))
        # 然后softmax,也就是让他们最后一个维度,和为1
        # attn:10,8,5,5
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # 这个attn就是计算出来的自注意力值，和v做点乘
        # attn:10,8,5,5
        # v:10,8,5,64
        # out:10,8,5,64
        out = rearrange(out, 'b h n d -> b n (h d)')
        # out:10,5,512
        out =  self.to_out(out)
        # 全连接,只管最后一层
        # out:10,5,512
        return out

def initialise_gaussian_weights(channels, kernel_size, sigma, dims):
    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dims
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dims

    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in kernel_size]
    )
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= (
            1
            / (std * math.sqrt(2 * math.pi))
            * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
        )

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
    return kernel


# https://github.com/masadcv/torchgaussianfilter
class GaussianFilter2d(nn.modules.Conv2d):
    def __init__(
        self,
        in_channels,
        kernel_size,
        sigma,
        padding="same",
        stride=1,
        padding_mode="zeros",
    ):
        gausssian_weights = initialise_gaussian_weights(
            channels=in_channels, kernel_size=kernel_size, sigma=sigma, dims=2
        )

        out_channels = gausssian_weights.shape[0]

        super(GaussianFilter2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=in_channels,
            bias=False,
            padding_mode=padding_mode,
        )

        # update weights
        # help from: https://discuss.pytorch.org/t/how-do-i-pass-numpy-array-to-conv2d-weight-for-initialization/56595/3
        with torch.no_grad():
            haar_weights = gausssian_weights.float().to(self.weight.device)
            self.weight.copy_(haar_weights)

# https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.padding = kernel_size//2
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, padding=self.padding ,weight=self.weight, groups=self.groups)


# smoothing = GaussianSmoothing(3, 5, 1)


# cv2.bilateralFilter
class BilateralFilter():
    def __init__(self) -> None:
        pass

    def __call__(self, batch:torch.Tensor):
        b = batch.size()[0]
        out = copy.copy(batch)
        to_tensor = transforms.ToTensor()
        for i in range(b):
            img = batch[i].numpy().transpose(1,2,0)
            for c in range(3):
                img[:,:,c] = cv2.bilateralFilter(img[:,:,c],0,50,10)
            out[i] = to_tensor(img)
        return out



# dct denoise

# from matplotlib import pyplot as plt
class DCTDenois():
    def __init__(self, alpha) -> None:
        self.mask = np.zeros((1,1))
        self.alpha = alpha

    def update_mask(self, h, w):
        h_now, w_now = self.mask.shape
        if h_now!=h or w_now!= w:
            self.mask = np.zeros((h,w))
            h1,w1 = int(h*self.alpha), int(w*self.alpha)
            ones = np.ones((h1,w1))
            self.mask[:h1,:w1] = ones

    def __call__(self, batch:torch.Tensor):
        b, _ ,h, w  = batch.size()
        self.update_mask(h, w)
        out = copy.copy(batch)
        to_tensor = transforms.ToTensor()
        for i in range(b):
            img = batch[i].numpy().transpose(1,2,0)
            for c in range(3):
                img_c = cv2.dct(img[:,:,c])
                img_c = img_c*self.mask
                img[:,:,c] = cv2.idct(img_c)
            out[i] = to_tensor(img)
        return out

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

        
class MultiFulterModel(nn.Module):
    def __init__(self, h, w) -> None:
        super(MultiFulterModel,self).__init__()
        
        self.h, self.w = h, w
        # self.self_att = Attention(512)
        self.self_att = nn.Conv2d(in_channels=18,out_channels=3,kernel_size=1)

    def __call__(self, x:torch.Tensor)->torch.Tensor:
        
        
        # x = torch.flatten(x, 2)
        x = self.self_att(x)
        # x = torch.mean(x,dim=1)
        # x = rearrange(x, 'b n (c h w) -> b n c h w', h=self.h, w=self.w)
        return x

class Loss():
    def __init__(self, norm_lambda=0.01):
        self.bce = nn.MSELoss()
        self.norm_lambda =norm_lambda

    def __call__(self, input:torch.Tensor, target):
        bceloss = self.bce(input, target)
        norm = self.norm_lambda * input.abs().mean()
        return bceloss + norm, bceloss, norm

class MultiFulter():
    def __init__(self, device=None, ckpt=None) -> None:
        input_size = [384, 384]
        print('model loaded')
        self.model = MultiFulterModel(*input_size)
        self.transform = transforms.Compose([
                transforms.Resize(size = input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.transform_eval = transforms.Compose([
                transforms.Resize(size = input_size),
                # transforms.Normalize([0.5], [0.5])
            ])
        self.device = device if device is not None else torch.device('cuda')
        # self.unnormalize = UnNormalize([0.5], [0.5], self.device)
        self.model = self.model.to(self.device)

        self.filters = []
        print('prepareing filters')
        self.filters.append(torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1))
        self.filters.append(torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.filters.append(MedianPool2d(kernel_size=3, same=True))
        self.filters.append(GaussianSmoothing(3, 5, 1))
        self.filters.append(BilateralFilter())
        self.filters.append(DCTDenois(alpha=0.5))
        print('filters prepared')

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
            lr_sch = optim.lr_scheduler.StepLR(optimizer, decay_step, decay_rate)
        elif sch == 'cos':
            lr_sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3, 2)
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

                filter_results = []
                for filter in self.filters:
                    filter_result = filter(inputs_noise)
                    filter_results.append(filter_result)
                # a = torch.chunk(x[0],3,1)
                # x = torch.stack(filter_results,dim=1)
                inputs_noise = torch.cat(filter_results, dim=1)
                inputs_noise = inputs_noise.to(self.device)

                # zero the grad
                optimizer.zero_grad()

                # forward
                result = self.model(inputs_noise)


                # loss and backward
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
            logger.info('====== epoch {} time: {:.2f} ETA: {:.2f} loss:{:.2f}======'.format(epoch+1, epoch_time, ETA, sum(epoch_loss)/len(epoch_loss)))
            lr_sch.step(epoch)

            # save model
            if epoch%decay_step==decay_step-1:
                #valid
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

                        filter_results = []
                        for filter in self.filters:
                            filter_result = filter(inputs_noise)
                            filter_results.append(filter_result)
                        inputs_noise = torch.cat(filter_results, dim=1)
                        inputs_noise = inputs_noise.to(self.device)
                        
                        optimizer.zero_grad()
                        result = self.model(inputs_noise)
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
                ckpt_name = os.path.join(save_path, 'dunet'+'_'+str(epoch+1)+'.pth')
                torch.save(ckpt_data, ckpt_name)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.model.eval()
            x = self.transform_eval(x)

            filter_results = []
            for filter in self.filters:
                filter_result = filter(x)
                filter_results.append(filter_result)
            x = torch.cat(filter_results, dim=1)
            x = x.to(self.device)

            output = self.model(x)
        return output
