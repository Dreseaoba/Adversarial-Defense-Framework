# -*- coding: utf-8 -*-
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from config_test import Config
from vision_image_folder import ImageFolder
from defense_model import ReconModel
import pdb


# parameters
#pdb.set_trace()
config = Config()

data_folder = config.data_folderGet()
# feat_dim = config.feat_dimGet()
# feat_path = config.feat_pathGet()
base_name = os.path.basename(data_folder)
save_folder = data_folder.replace(base_name, base_name+'PIN')
os.makedirs(save_folder, exist_ok=True)

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
model = ReconModel(config)
model = model.cuda()

# pre-process
transforms_func = transforms.Compose([
        transforms.Resize(size = [112,112]),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    
# get data
data_set = ImageFolder(data_folder, transform=transforms_func)
data_loader = DataLoader(data_set, batch_size=1, shuffle=False)


# feature extraction
N = len(data_set)
# features = torch.zeros(N, feat_dim)

for i, data in enumerate(data_loader, 0):
    # input data
    inputs, names, _ = data
    inputs = Variable(inputs)
    inputs = inputs.cuda()
    
    # forward
    x_recon = model(inputs)
    
    x_recon = x_recon.detach().cpu()
    x_recon = unnormalize(x_recon).squeeze()
    # print(x_recon, type(x_recon), x_recon.shape, names[0])
    toPIL = transforms.ToPILImage()
    img = toPIL(x_recon)
    img.save(os.path.join(save_folder, names[0]))
    
# torch.save(features, feat_path)
    
    
