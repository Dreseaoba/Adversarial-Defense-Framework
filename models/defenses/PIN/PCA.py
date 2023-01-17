from datetime import datetime
import time
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
#from torchvision.datasets import ImageFolder
from torchvision import transforms

from config_train import Config
from vision_image_folder import ImageFolder


config = Config()

data_folder = config.data_folderGet()
transform = transforms.Compose([
                transforms.Resize(size = [112,112]),
                transforms.Grayscale(1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
data_set = ImageFolder(data_folder, transform=transform)
data_loader = DataLoader(data_set, batch_size=40000, shuffle=False, num_workers=8)

for i, data in enumerate(data_loader, 0):
    # input data
    inputs, _, _ = data
    inputs = inputs.cuda()
    print(inputs.size())
    inputs = inputs.view(inputs.size()[0], -1)
    print(inputs.size())
    n = inputs.size()[0]
    mean = torch.mean(inputs, axis=0)
    print('mean', mean.size())
    inputs = inputs - mean
    covariance_matrix = 1 / n * torch.matmul(inputs.T, inputs)
    print('cov', covariance_matrix.size())
    eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
    eigenvalues = torch.norm(eigenvalues, dim=1)
    idx = torch.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    print('eigenvectors', eigenvectors.size())
    state = {'eig_vec':eigenvectors, 'avg_face':mean}
    torch.save(state, '/data/guohaoxun/workspace/atk/AdvDefenseFramework/models/defenses/PIN/face_eigen/dpfk32_eig.pth')
