# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:48:36 2018

@author: 60236
"""

from model import resnet50
from torchvision.utils import save_image
import torch
from datasets import test_data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from datasets import datagen
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate of progress')
parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')
parser.add_argument('--img_size', type=int, default=650, help='size of each image dimension')
parser.add_argument('--root', type=str, default=r'./shuffle_txt.txt', help='path to images file')
args = parser.parse_args()
print(args)


transform = [  # transforms.Resize((args.img_size,args.img_size),interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ]

os.makedirs('./output', exist_ok=True)

net = resnet50()
checkpoint = torch.load('./weights/ckpt.pth')
net.load_state_dict(checkpoint['weights'])
print('loading weights success...')
net = net.cuda()
net.eval()

test_ratio = 0.1
test  = datagen(args.root,transform,transform, train=False, test_ratio=test_ratio)
test_loader = DataLoader(test, batch_size=args.batch_size)

print('the total numbers of testsets is: ', len(test))

src, tar = [], []
with open(args.root,'r') as f:
    lines = f.readlines()
    for line in lines:
        a, b = line.strip().split(' ')
        src.append(a)
        tar.append(b)
   
    num_test = len(src) * test_ratio * (-1)
    num_test = int(num_test)
    
    src = src[num_test:]
    tar = tar[num_test:]

image_shotname = []

for path in src:
    head, tail = path.split('.')
    start = head.find('img')+3
    name = head[start:] 
    image_shotname.append(name)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for j, img in enumerate(test_loader):
    sr = img['sr'].to(device)
    label = img['ta'].to(device)
    result = net(sr)
    
#    cat = torch.cat((input_im,result),0)
    cat = result
    save_image(cat,'./output/seg_result_{}.png'.format(image_shotname[j]))
    print('\nsaving image ...',j)
