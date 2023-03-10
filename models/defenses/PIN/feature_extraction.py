# -*- coding: utf-8 -*-
"""
Feature extraction by AID
RenMin
"""

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from config_test import Config
from vision_image_folder import ImageFolder
from defense_model import DefenseModel
import pdb


# parameters
#pdb.set_trace()
config = Config()

data_folder = config.data_folderGet()
feat_dim = config.feat_dimGet()
feat_path = config.feat_pathGet()

# model
model = DefenseModel(config)
model = model.cuda()

# pre-process
transforms = transforms.Compose([
        transforms.Resize(size = [112,112]),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    
# get data
data_set = ImageFolder(data_folder, transform=transforms)
data_loader = DataLoader(data_set, batch_size=1, shuffle=False)


# feature extraction
N = len(data_set)
features = torch.zeros(N, 1, feat_dim)
names = []
for i, data in enumerate(data_loader, 0):
    # input data
    inputs, name, _ = data
    inputs = Variable(inputs)
    inputs = inputs.cuda()
    
    # forward
    feat = model(inputs)
    
    features[i, :] = feat
    names.append(name)
criterion_cos = torch.nn.CosineSimilarity()    
for i in range(N):
    for j in range(i+1,N):
        print(names[i],names[j],criterion_cos(features[i], features[j]).detach().cpu().numpy()[0])

# torch.save(features, feat_path)
    
    
