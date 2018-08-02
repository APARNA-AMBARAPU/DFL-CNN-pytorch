# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as T
from torch.utils.data.dataloader import default_collate
import datetime
import sys
f_result=open('/home/pixiym/chb/DFL-CNN/result1.txt', 'w') 
sys.stdout=f_result


time_stamp = datetime.datetime.now()
print("time_stamp_start       " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
# Hyper parameters
num_epochs = 1000
batch_size = 5
learning_rate = 1

#dataset
transform = T.Compose([
    T.Resize(224), # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    T.CenterCrop(224), # 从图片中间切出224*224的图片
    T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
])

class custom_dset(Dataset):
    def __init__(self,
                 img_path,
                 txt_path,
                 img_transform=None):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.img_list = [
                os.path.join(img_path, i.split()[0]) for i in lines
            ]
            self.label_list = [float(i.split()[1]) for i in lines]
        self.img_transform = img_transform
        
    

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        img = Image.open(img_path).convert('RGB').resize((224, 224)) 
        if self.img_transform is not None:
            img = self.img_transform(img)
        
        return img, label

    def __len__(self):
        return len(self.label_list)

train_data = custom_dset( "/media/pixiym/CVPR/images", "/media/pixiym/CVPR/images/train_images.txt",img_transform=transform)



test_data = custom_dset( "/media/pixiym/CVPR/images", "/media/pixiym/CVPR/images/test_images.txt",img_transform=transform)


#Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,collate_fn=default_collate,drop_last=True, pin_memory=True,
                                           shuffle=True,num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,collate_fn=default_collate,pin_memory=True,
                                          shuffle=False,num_workers=4)


time_stamp = datetime.datetime.now()
print("time_stamp_data       " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S')) 


#vgg16 conv1--conv4
vgg16featuremap = torchvision.models.vgg16(pretrained=True).features
conv1_conv4 = torch.nn.Sequential(*list(vgg16featuremap.children())[:-8])
#P_stream conv6
k = 512
m = 200
conv6 = torch.nn.Conv2d(512, k*m, kernel_size=1, stride=1, padding=0)
#P_stream pool6
pool6 = torch.nn.MaxPool2d((28, 28), stride=(28, 28))
#G-stream con5
conv5 = torch.nn.Sequential(*list(vgg16featuremap.children())[-8:])
#G-Stream module
class G_Stream_net(nn.Module):
    def __init__(self):
        super(G_Stream_net, self).__init__()
        self.conv1_conv4 = conv1_conv4
        self.conv5 = conv5 
        self.fc =  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 200),
        )
    def forward(self, x):
        out = self.conv1_conv4(x)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

G_Stream_net = G_Stream_net().cuda()

#P-Stream module

class P_Stream_net(nn.Module):
    def __init__(self):
        super(P_Stream_net, self).__init__()
        self.conv1_conv4 = conv1_conv4
        self.conv6 = conv6 
        self.pool6 = pool6
        self.fc = nn.Linear(k*m, 200)
    def forward(self, x):
        out = self.conv1_conv4(x)
        out = self.conv6(out)
        out = self.pool6(out)
        # L2 normalization
        out = F.normalize(out, p=2, dim=1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
P_Stream_net = P_Stream_net().cuda()

#Side Branch Module
class Side_Branch_net(nn.Module):
    def __init__(self):
        super(Side_Branch_net, self).__init__()
        self.conv1_conv4 = conv1_conv4
        self.conv6 = conv6 
        self.pool6 = pool6
        self.AvgPool1d = nn.AvgPool1d(k, stride=k)
    def forward(self, x):
        out = self.conv1_conv4(x)
        out = self.conv6(out)
        out = self.pool6(out)
        N1 = out.size()[0]
        out = out.view(N1,-1,k*m)
        out = self.AvgPool1d(out)
        # L2 normalization
        out = out.view(N1,m,1,1)
        out = F.normalize(out, p=2, dim=1)
        return out
Side_Branch_net = Side_Branch_net().cuda()


criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.long().cuda()
        # Forward pass
        outputs1 = G_Stream_net(images)
        N1 =  outputs1.size()[0]
        N2 =  outputs1.size()[1]
        outputs2 = P_Stream_net(images)
        outputs3 = Side_Branch_net(images)
        outputs3 = outputs3.view(N1,N2)
        outputs = 0.6*outputs1 + 0.2*outputs2 + 0.2*outputs3
        loss = criterion(outputs, labels)
        
        print(outputs1.size())
        print(outputs2.size())
        print(outputs3.size())
        print(outputs.size())
        print(labels.size())
        print(loss.size())
