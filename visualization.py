import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import torch
import random

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
import pickle
from torchsummary import summary
from types import SimpleNamespace
warnings.filterwarnings("ignore")


def plot_all_accuracy():
    k_choice = [1,3,5,10,20]
    MLP = [0.6144, 0.6326, 0.6419, 0.6469, 0.6454]
    VGG = [0.134, 0.137,0.135, 0.134, 0.133]
    Resnet = [0.776, 0.798, 0.807, 0.818, 0.826]
    CNN = [0.686, 0.752, 0.783, 0.813, 0.823]
    Inception = [0.812, 0.852, 0.863, 0.873, 0.881]
    CNNCL =[0.797, 0.8221, 0.8226, 0.8441, 0.8407]

    fig, axs = plt.subplots(1,1,figsize=(5,5))
    axs.plot(k_choice,MLP,label='MLP')
    axs.plot(k_choice,VGG,label='VGG')
    axs.plot(k_choice,Resnet,label='Resnet')
    axs.plot(k_choice,CNN,label='CNN')
    axs.plot(k_choice,Inception,label='Inception')
    axs.plot(k_choice,CNNCL,label='CNNCL')


    axs.set_xlabel('K Value')
    axs.set_ylabel('Search Accuracy')
    axs.legend()
    fig.show()



def load_model_history(filepath):
    with open(filepath, 'rb') as file:
        metrics = pickle.load(file)
    return metrics

def plot_model_history(model_history, epoch_num = 5, plot_figure = False, txt_log = False, search_result = False, Is_CNNCL = False):
    train_realworld_loss = model_history['train_realworld_loss']
    val_realworld_loss = model_history['val_realworld_loss']
    train_quickdraw_loss = model_history['train_quickdraw_loss']
    val_quickdraw_loss = model_history['val_quickdraw_loss']
    
    train_realworld_acc = model_history['train_realworld_acc']
    val_realworld_acc = model_history['val_realworld_acc']
    train_quickdraw_acc = model_history['train_quickdraw_acc']
    val_quickdraw_acc = model_history['val_quickdraw_acc']
    k_choice = model_history['k_choice']
    topk_acc = model_history['topk_acc']
    
    if plot_figure == True:
        fig, axs = plt.subplots(2,2,figsize=(10,10))
        axs[0,0].plot(model_history['epoch'], train_realworld_loss, label = 'realworld train loss')
        axs[0,0].plot(model_history['epoch'], val_realworld_loss, label = 'realworld val loss')
        axs[0,1].plot(model_history['epoch'], train_quickdraw_loss, label = 'quickdraw train loss')
        axs[0,1].plot(model_history['epoch'], val_quickdraw_loss, label = 'quickdraw val loss')

        axs[1,0].plot(model_history['epoch'], train_realworld_acc, label = 'realworld train accuracy')
        axs[1,0].plot(model_history['epoch'], val_realworld_acc, label = 'realworld val accuracy')
        axs[1,1].plot(model_history['epoch'], train_quickdraw_acc, label = 'quickdraw train accuracy')
        axs[1,1].plot(model_history['epoch'], val_quickdraw_acc, label = 'quickdraw val accuracy')

        axs[0,0].set_xlabel('Epoch')
        axs[0,0].set_ylabel('Loss')
        axs[0,1].set_xlabel('Epoch')
        axs[0,1].set_ylabel('Loss')
        axs[1,0].set_xlabel('Epoch')
        axs[1,0].set_ylabel('Accuracy')
        axs[1,1].set_xlabel('Epoch')
        axs[1,1].set_ylabel('Accuracy')
        
        axs[0,0].legend()
        axs[0,1].legend()
        axs[1,0].legend()
        axs[1,1].legend()
        fig.show()
        
    if txt_log == True:
        output_epoch = min(len(model_history['epoch']), epoch_num)
        print(f'--------------------------------------- Real-World Model --------------------------------------')
        for i in range(output_epoch):
            print(f'Epoch {i:^3}: Train Loss: {train_realworld_loss[i]:.4f}| Val Loss: {(val_realworld_loss[i]):.4f}| Train Accuracy: {(train_realworld_acc[i]):.4f}| Val Accuracy: {(val_realworld_acc[i]):.4f}')
        print('\n')
        print(f'--------------------------------------- QuickDraw Model --------------------------------------')
        for i in range(output_epoch):
            print(f'Epoch {i:^3}: Train Loss: {train_realworld_loss[i]:.4f}| Val Loss: {(val_realworld_loss[i]):.4f}| Train Accuracy: {(train_quickdraw_acc[i]):.4f}| Val Accuracy: {(val_quickdraw_acc[i]):.4f}')
            
    if search_result == True:
        print(f'--------------------------------------- Search Engine Result --------------------------------------')
        for i in range(len(k_choice)):
            print(f'Find Top {k_choice[i]:^3} Similar Real-world Images for One Sketch: Accuracy: {topk_acc[i]:.4f}')
        print('\n')
        fig, axs = plt.subplots(1,1,figsize=(5,5))
        axs.plot(k_choice,topk_acc)
        axs.set_xlabel('K Value')
        axs.set_ylabel('Search Accuracy')
        fig.show()
        
class MLP(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(x_dim, 1024, bias=False),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(1024, 512, bias=False),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(256, 128, bias=False),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(128, y_dim, bias=False)
        )
        
    def forward(self, x):
        x = self.mlp(x.flatten(1))
        return x 
    
def convbn(in_channels, out_channels, kernel_size, stride, padding, bias):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class CNN(nn.Module):
    def __init__(self, n_channels=3, n_classes=7, dropout=0.1):
        super(CNN, self).__init__()
        layer1   = convbn(n_channels, 64, kernel_size=3, stride=2, padding=1, bias=True)
        layer2   = convbn(64,  128, kernel_size=3, stride=2, padding=1, bias=True)
        layer3   = convbn(128, 192, kernel_size=3, stride=2, padding=1, bias=True)
        layer4   = convbn(192, 256, kernel_size=3, stride=2, padding=1, bias=True)
        layer1_2 = convbn(64,  64,  kernel_size=3, stride=1, padding=0, bias=True)
        layer2_2 = convbn(128, 128, kernel_size=3, stride=1, padding=0, bias=True)
        layer3_2 = convbn(192, 192, kernel_size=3, stride=1, padding=0, bias=True)
        layer4_2 = convbn(256, 256, kernel_size=3, stride=1, padding=0, bias=True)
        
        pool = nn.AdaptiveAvgPool2d((1,1))
        self.layers = nn.Sequential(layer1, layer1_2, layer2, layer2_2, layer3, layer3_2, layer4, layer4_2, pool)
        self.nn = nn.Linear(256, n_classes)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        feats = self.layers(x).flatten(1)
        x = self.nn(self.dropout(feats))
        return x, feats
    
class CNN2(nn.Module):
    def __init__(self, n_channels=3, n_classes=7, dropout=0.1):
        super(CNN2, self).__init__()
        layer1   = convbn(n_channels, 64, kernel_size=3, stride=2, padding=1, bias=True)
        layer2   = convbn(64,  128, kernel_size=3, stride=2, padding=1, bias=True)
        layer3   = convbn(128, 256, kernel_size=3, stride=2, padding=1, bias=True)
        layer1_2 = convbn(64,  64,  kernel_size=3, stride=1, padding=0, bias=True)
        layer2_2 = convbn(128, 128, kernel_size=3, stride=1, padding=0, bias=True)
        layer3_2 = convbn(256, 256, kernel_size=3, stride=1, padding=0, bias=True)
        pool = nn.AdaptiveAvgPool2d((1,1))
        self.layers = nn.Sequential(layer1, layer1_2, layer2, layer2_2, layer3, layer3_2, pool)
        self.nn = nn.Linear(256, n_classes)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        feats = self.layers(x).flatten(1)
        x = self.nn(self.dropout(feats))
        return x, feats

class CNNCL(nn.Module):
    def __init__(self, n_classes=7, dropout=0.1, t=0.1):
        super(CNNCL, self).__init__()
        self.t = t
        self.quickdraw_model = CNN()
        self.realworld_model = CNN2()
    def forward(self, quickdraw_x, realworld_x):
        quickdraw_pred, quickdraw_feat = self.quickdraw_model(quickdraw_x)
        realworld_pred, realworld_feat = self.realworld_model(realworld_x)
        return quickdraw_pred, quickdraw_feat, realworld_pred, realworld_feat
    
class VGG16(nn.Module):
    def __init__(self, num_classes=7):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2*2*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
      
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
'''
Now, that we have created the ResidualBlock, we can build our ResNet.

there are three blocks in the architecture,
To make this block, we create a helper function _make_layer. 
The function adds the layers one by one along with the Residual Block. 
After the blocks, we add the average pooling and the final linear layer.
'''

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 7):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
class CNN_original(nn.Module):
    
    def __init__(self, num_classes):
        super(CNN_original, self).__init__()
       
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.flatten = nn.Flatten(start_dim=1, end_dim=3)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(128, 7)

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = self.dropout(x)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6(x)), (2, 2))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = F.adaptive_avg_pool2d(F.relu(self.conv8(x)), 1)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dropout(x)
        out = self.fc1(x)
        return out
    
    
act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}

class InceptionBlock(nn.Module):

    def __init__(self, c_in, c_red : dict, c_out : dict, act_fn):
        super().__init__()

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            nn.BatchNorm2d(c_out["1x1"]),
            act_fn()
        )

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn()
        )

        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn()
        )

        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn()
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out

class Inception(nn.Module):

    def __init__(self, num_classes=7, act_fn_name="relu", **kwargs):
        super().__init__()
        self.hparams = SimpleNamespace(num_classes=num_classes,
                                       act_fn_name=act_fn_name,
                                       act_fn=act_fn_by_name[act_fn_name])
        self._create_network()
        self._init_params()

    def _create_network(self):
        self.input_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.hparams.act_fn()
        )
        self.inception_blocks = nn.Sequential(
            InceptionBlock(64, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8}, act_fn=self.hparams.act_fn),
            InceptionBlock(64, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}, act_fn=self.hparams.act_fn),
            nn.MaxPool2d(3, stride=2, padding=1),
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}, act_fn=self.hparams.act_fn),
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}, act_fn=self.hparams.act_fn),
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}, act_fn=self.hparams.act_fn),
            InceptionBlock(96, c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24}, act_fn=self.hparams.act_fn),
            nn.MaxPool2d(3, stride=2, padding=1),
            InceptionBlock(128, c_red={"3x3": 48, "5x5": 16}, c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}, act_fn=self.hparams.act_fn),
            InceptionBlock(128, c_red={"3x3": 48, "5x5": 16}, c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}, act_fn=self.hparams.act_fn)
        )
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.hparams.num_classes)
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)
        return x