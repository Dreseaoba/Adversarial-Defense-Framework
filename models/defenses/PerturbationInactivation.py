# -*- coding: utf-8 -*-
"""
train pca selection model with policy gradient
RenMin 20190918
"""

from datetime import datetime
import time
import os
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
#from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F

from models.defenses.PIN.config_train import Config
from models.defenses.PIN.vision_image_folder import ImageFolder
from models.defenses.PIN.pca_select_model import Encoder, PCASelection, DAE
import models.defenses.PIN.pca_loss_fn as PLF
import models.defenses.PIN.recon_fn as RF


# import pdb


# parameters
#pdb.set_trace()
config = Config()

# EPOCHES = config.epochGet()
# BATCH = config.batchGet()
LR_en = config.lr_encoderGet()
LR_pca = config.lr_pcafcGet()
MOMENTUM = config.momentumGet()
NOISE_SCALE = config.noise_scaleGet()


Lamb = config.lamb_sparseGet()
lambda_mean = config.lamb_meanGet()
eigen_path = config.eigen_faceGet()

decay_step = config.decay_stepGet()
decay_rate = config.decay_rateGet()
dae_path = config.ckpt_daeGet()

data_folder = config.data_folderGet()
# save_step = config.save_stepGet()
encoder_path = config.encoder_pathGet()
pcafc_path = config.pcafc_pathGet()
en_ckpt = config.ckpt_encoderGet()
pca_ckpt = config.ckpt_PCAfcGet()




class PerturbationInactivation():
    def __init__(self, mode='eval', devices=None):
        """PerturbationInactivation

        Args:
            `logger` (Logger)
            `mode` (str): 'eval' or 'train'
            `devices` (tuple of int, default None): two ids of cuda devices, one for mdoel and one for egi
        """
        print('INITIALIZING...')
        self.encoder = Encoder()
        self.pca_layer = PCASelection()
        self.dae = DAE()
        self.devices = devices if devices is not None else [0,1]
        self.devices = [torch.device('cuda:{}'.format(_d)) for _d in self.devices]
        
        if mode=='eval':
            en_data = torch.load(en_ckpt, map_location=lambda storage, loc:storage)
            self.encoder.load_state_dict(en_data['model'])
            pca_data = torch.load(pca_ckpt, map_location=lambda storage, loc:storage)
            self.pca_layer.load_state_dict(pca_data['model'])


        self.encoder = self.encoder.to(self.devices[0])
        self.pca_layer = self.pca_layer.to(self.devices[0])
        dae_data = torch.load(dae_path, map_location=lambda storage, loc:storage)
        self.dae.load_state_dict(dae_data['model'])
        self.dae = self.dae.to(self.devices[0])
        self.recon_fn = RF.EigRecon_Act(config.eigen_faceGet())

        # optimizer
        params = []
        for name, value in self.encoder.named_parameters():
            params += [{'params':value, 'lr':LR_en}]
        for name, value in self.pca_layer.named_parameters():
            params += [{'params':value, 'lr':LR_pca}]

        self.optimizer = optim.SGD(params=params, lr=LR_en, momentum=MOMENTUM)
        self.lr_sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 3, 2)
        # self.lr_sch = optim.lr_scheduler.StepLR(self.optimizer, decay_step, decay_rate)

        # pre-process
        self.transform = transforms.Compose([
                transforms.Resize(size = [112,112]),
                transforms.Grayscale(1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

        self.transform_eval = transforms.Compose([
                transforms.Resize(size = [112,112]),
                transforms.Grayscale(1),
                transforms.Normalize([0.5], [0.5])
            ])
    
    def noise_fn(self, inputs: torch.Tensor, scale: float):
        """
        add Gaussion noise to inputs
        sacle is the Sigmma of Gaission distribution
        """
        noise = torch.randn(inputs.size())
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
        noise_path=None):
        # get data
        print('LOADING DATA')
        data_set = ImageFolder(data_folder, transform=self.transform)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
        print(len(data_loader))

        # loss function
        mc_loss = PLF.MCLoss(Lamb, eigen_path, self.devices[1])

        #train
        #pdb.set_trace()
        self.dae.eval()
        print('START TRAINING')
        last_epoch_time = 0
        for epoch in range(max_epoch):
            logger.info('====== epoch {} ======'.format(epoch+1))
            epoch_time = time.time()
            running_loss = 0.
            self.encoder.train()
            self.pca_layer.train()
            for i, data in enumerate(data_loader, 0):
                # input data
                inputs, _, _ = data
                inputs = Variable(inputs)
                inputs_noise = self.noise_fn(inputs, NOISE_SCALE)
                inputs = inputs.to(self.devices[0])
                inputs_noise = inputs_noise.to(self.devices[0])

                # zero the grad
                self.optimizer.zero_grad()

                # forward
                _, inputs_recon = self.dae(inputs_noise)
                hidden = self.encoder(inputs_recon)
                recon = self.pca_layer(hidden)

                # loss and backward
                loss, mse_loss = mc_loss(
                    recon.to(self.devices[1]), 
                    inputs.to(self.devices[1]), 
                    inputs_noise.to(self.devices[1]))
                loss = loss.to(self.devices[0])
                loss.backward()
                # loss.backward()
                self.optimizer.step()

                # print log
                running_loss += float(loss.item())
                # running_loss += float(loss.item())
                if (i+1)%3==0:
                    logger.info('epoch: {}, step: {}, loss: {}'.format(epoch+1, i+1, running_loss/1000.))
                    running_loss = 0.
            epoch_time = time.time() - epoch_time
            if last_epoch_time==0:
                ETA = (max_epoch - epoch - 1) * epoch_time
            else:
                ETA = (max_epoch - epoch - 1) * (epoch_time + last_epoch_time)/2
            logger.info('====== epoch {} time: {:.2f} ETA: {:.2f} ======'.format(epoch+1, epoch_time, ETA))
            self.lr_sch.step(epoch)

            # save model
            if epoch%decay_step==decay_step-1:
                en_data = dict(
                        optimizer = self.optimizer.state_dict(),
                        model = self.encoder.state_dict(),
                        epoches = epoch+1,
                        )
                en_name = encoder_path+'_'+str(epoch+1)+'.pth'
                torch.save(en_data, en_name)

                pca_data = dict(
                        model = self.pca_layer.state_dict(),
                        epoches = epoch+1,
                        )
                pca_name = pcafc_path+'_'+str(epoch+1)+'.pth'
                torch.save(pca_data, pca_name)

    def get_pca_act(self, x):
        uni_dis = torch.rand(x.size()).cuda()
        pca_act = x>uni_dis
        return pca_act
    
    def get_pca_act_fix(self, x):
        pca_act = x>x.mean()
        return pca_act

    def proc_before_reg(self, x):
        x = x.expand(x.size(0),3,x.size(2),x.size(3))
        return x
    
    
    def __call__(self, x_raw):
        class UnNormalize:
            #restore from T.Normalize
            #反归一化
            def __init__(self,mean=(0.485, 0.456, 0.406),std= (0.229, 0.224, 0.225)):
                self.mean=torch.tensor(mean).view((1,-1,1,1))
                self.std=torch.tensor(std).view((1,-1,1,1))
            def __call__(self,x):
                x=(x*self.std)+self.mean
                return torch.clip(x,0,None)

        unnormalize = UnNormalize()

        # model
        self.dae.eval()
        self.encoder.eval()
        self.pca_layer.eval()
        # self.reg_model.eval()
        
        x_raw = x_raw.to(self.devices[0])
        x_raw = self.transform_eval(x_raw)
        _, x = self.dae(x_raw)
        x = self.encoder(x)
        x = self.pca_layer(x)
        pca_act = self.get_pca_act(x)
        x_recon = self.recon_fn.recon(x_raw, pca_act)
        x_recon = self.proc_before_reg(x_recon)
        
        x_recon = x_recon.detach().cpu()
        x_recon = unnormalize(x_recon)
        # print(x_recon, type(x_recon), x_recon.shape, names[0])
        # toPIL = transforms.ToPILImage()
        # img = toPIL(x_recon[0])
        # img.save('/data/guohaoxun/workspace/atk/AdvDefenseFramework/tmp/pin.png')
        return x_recon
            



