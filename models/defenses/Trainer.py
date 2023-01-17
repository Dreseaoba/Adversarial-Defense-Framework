import logging
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair, _quadruple
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
from models.defenses.PIN.vision_image_folder import ImageFolder
from models.defenses.dataset import AdvDataset
import time
import torch.optim as optim
import os



class Trainer():
    def __init__(self, 
                model:nn.Module,
                transform,
                data_path,
                batch_size,
                loss,
                max_epoch,
                save_path,
                lr=0.01,
                curr_epoch=0,
                sch='step',
                momentum=0,
                decay_step=2,
                decay_rate=0.5,
                noise_scale=0.04,
                noise_path=None,
                valid_path=None,
                ) -> None:
        self.model = model
        self.loss = loss

        self.noise_training = noise_path is None
        self.valid = valid_path is not None
        print('LOADING DATA')
        if self.noise_training:
            data_set = ImageFolder(data_path, transform=transform)
        else:
            data_set = AdvDataset(data_path, noise_path, transform=transform)
        self.data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
        if self.valid:
            valid_set = AdvDataset(os.path.join(valid_path, 'ori'), os.path.join(valid_path, 'adv'), transform=transform)
            self.valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        print(len(self.data_loader))

        self.optimizer = optim.SGD(params=self.model.parameters(), lr=lr, momentum=momentum)
        if sch == 'step':
            self.lr_sch = optim.lr_scheduler.StepLR(self.optimizer, decay_step, decay_rate)
        elif sch == 'cos':
            self.lr_sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 3, 2)
        else:
            raise NotImplementedError('no scheduler named {}'.format(sch))

        self.curr_epoch = curr_epoch
        self.max_epoch = max_epoch
        self.noise_scale = noise_scale
        self.decay_step = decay_step
        self.save_path = save_path

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


    def run(self,logger:logging.Logger, device):
        
        self.model.train()

        print('START TRAINING')
        last_epoch_time = 0
        
        for epoch in range(self.curr_epoch, self.max_epoch):
            logger.info('====== epoch {}/{} ======'.format(epoch+1, self.max_epoch))
            epoch_time = time.time()
            running_loss = 0.
            epoch_loss = []
            for i, data in enumerate(self.data_loader, 0):
                # input data
                if self.noise_training:
                    inputs, _, _ = data
                    inputs = Variable(inputs)
                    inputs_noise = self.noise_fn(inputs, self.noise_scale)
                else:
                    inputs_noise, inputs = data
                inputs = inputs.to(device)

                filter_results = []
                for filter in self.filters:
                    filter_result = filter(inputs_noise)
                    filter_results.append(filter_result)
                # a = torch.chunk(x[0],3,1)
                # x = torch.stack(filter_results,dim=1)
                inputs_noise = torch.cat(filter_results, dim=1)
                inputs_noise = inputs_noise.to(device)

                # zero the grad
                self.optimizer.zero_grad()

                # forward
                result = self.model(inputs_noise)


                # loss and backward
                loss, _, norm = self.loss(result, inputs)
                loss = loss.to(device)
                loss.backward()
                # loss.backward()
                self.optimizer.step()

                # print log
                running_loss += float(loss.item())
                # running_loss += float(loss.item())
                if (i+1)%100==0:
                    logger.info('epoch: {}, step: {}, loss: {}'.format(epoch+1, i+1, running_loss))
                    epoch_loss.append(running_loss)
                    running_loss = 0.

            
            epoch_time = time.time() - epoch_time
            if last_epoch_time==0:
                ETA = (self.max_epoch - epoch - 1) * epoch_time
            else:
                ETA = (self.max_epoch - epoch - 1) * (epoch_time + last_epoch_time)/2
            logger.info('====== epoch {} time: {:.2f} ETA: {:.2f} loss:{:.2f}======'.format(epoch+1, epoch_time, ETA, sum(epoch_loss)/len(epoch_loss)))
            self.lr_sch.step(epoch)

            # save model
            if epoch%self.decay_step==self.decay_step-1:
                #valid
                if self.valid:
                    valid_loss = []
                    for i, data in enumerate(self.valid_loader, 0):
                        if self.noise_training:
                            inputs, _, _ = data
                            inputs = Variable(inputs)
                            inputs_noise = self.noise_fn(inputs, self.noise_scale)
                        else:
                            inputs_noise, inputs = data
                        inputs = inputs.to(device)

                        filter_results = []
                        for filter in self.filters:
                            filter_result = filter(inputs_noise)
                            filter_results.append(filter_result)
                        inputs_noise = torch.cat(filter_results, dim=1)
                        inputs_noise = inputs_noise.to(device)
                        
                        self.optimizer.zero_grad()
                        result = self.model(inputs_noise)
                        loss, _, norm = self.loss(result, inputs)
                        loss = loss.to(device)
                        running_loss += float(loss.item())
                        if (i+1)%100==0:
                            valid_loss.append(running_loss)
                            running_loss = 0.
                    logger.info('====== epoch {} valid loss:{:.2f}======'.format(epoch+1, sum(valid_loss)/len(valid_loss)))
                ckpt_data = dict(
                        optimizer = self.optimizer.state_dict(),
                        model = self.model.state_dict(),
                        epoches = epoch+1,
                        )
                ckpt_name = os.path.join(self.save_path, 'dunet'+'_'+str(epoch+1)+'.pth')
                torch.save(ckpt_data, ckpt_name)