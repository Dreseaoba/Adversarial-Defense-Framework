# import imp
from statistics import mode
import torch
from arcface_model import Backbone
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from config_test import Config
from vision_image_folder import ImageFolder
import pdb
import os
from PIL import Image

config = Config()

data_folder = config.data_folderGet()
# feat_dim = config.feat_dimGet()
# feat_path = config.feat_pathGet()




# model
model = Backbone(50, 0.4, 'ir_se')
model = model.cuda()
model.eval()

# pre-process
transforms_func = transforms.Compose([
        transforms.Resize(size = [112,112]),
        # transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    
# get data
# data_set = ImageFolder(data_folder, transform=transforms_func)
# data_loader = DataLoader(data_set, batch_size=1, shuffle=False)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def load_image(path):
        # name = os.path.basename(path)
        # path, target = self.samples[index]
        sample = pil_loader(path)
        sample = transforms_func(sample)
        sample = sample[None, :]
        # sample = sample.expand(sample.size(0),3,sample.size(2),sample.size(3))
        return sample

# recognition criterion
criterion_cos = torch.nn.CosineSimilarity()

src_path = '/data/guohaoxun/workspace/atk/GenAP/datasets/anchor_face_aligned/1000/1000_059563.jpg'
# src_path = '/data/guohaoxun/workspace/atk/GenAP/attack_results/2022-08-29_14:25/merge/source_1000_target_1012.png'
tgt_path = '/data/guohaoxun/workspace/atk/GenAP/datasets/anchor_face_aligned/1012/1012_118475.jpg'
src_img = load_image(src_path)
tgt_img = load_image(tgt_path)


src = Variable(src_img)
src = src.cuda()
src_feat = model(src)

tgt = Variable(tgt_img)
tgt = tgt.cuda()
target_feat = model(tgt)

sim = criterion_cos(src_feat, target_feat).detach().cpu().numpy()[0]
same = sim > 0.28402677178382874
print(same, sim)








