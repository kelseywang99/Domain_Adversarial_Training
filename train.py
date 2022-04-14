#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


### 模型搭建和训练

# ### 1、构建模型
# 有特征提取器、标签分类器、Domain分类器三个部分。\
# FeatrueExtractor采用卷积网络，输入图像来提取特征；\
# LabelPredictor采用全连接网络，输入feature extractor得到的feature来判断动物类别；\
# DomainClassifier采用全连接网络，输入feature extractor得到的feature来判断feature是来自source data还是target data。
class FeatureExtractor(nn.Module):
    # 卷积网络
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), 

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), 

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), 

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), 

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x


class LabelPredictor(nn.Module):
    # 全连接
    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512*2*2, 512),
            nn.ReLU(),
            nn.Dropout(p = 0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.2),

            nn.Linear(256, 9)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DomainClassifier(nn.Module):
    # 全连接
    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512*2*2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ### 2、训练模型
def train_epoch(source_dataloader, target_dataloader, lamb):
    # lamb: adversarial layer的loss系数

    # running_D_loss: DomainClassifier的loss
    # running_F_loss: FeatureExtrator 和 LabelPredictor 的loss
    running_D_loss, running_F_loss = 0.0, 0.0
    # correct: 分类正确的个数 
    # total_num: source data总共的个数
    correct, total_num = 0.0, 0.0

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        # 提取data
        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        
        # source data和target data分布不同，mean和var不同，batch_norm 的时候会出问题
        # 所以要拼接在一起 
        mixed_data = torch.cat([source_data, target_data], dim=0)
        # 设置domain_label，source是1，target是0
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        domain_label[:source_data.shape[0]] = 1

        # 训练Domain Classifier
        feature = feature_extractor(mixed_data)
        # 固定Feature Extractor，不要梯度下降
        domain_logits = domain_classifier(feature.detach())
        # 计算domain classifier的loss
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss+= loss.item()
        # 反向传播，domain classifier希望正确分类，对domain classifier梯度下降
        loss.backward()
        optimizer_D.step()

        # 训练Feature Extractor和Label Predictor
        # source data经过feature extractor后的feature作为input，用label predictor预测label
        class_logits = label_predictor(feature[:source_data.shape[0]])
        # source data经过feature extractor后的feature作为input，用domain classifier预测domain label
        domain_logits = domain_classifier(feature)
        # loss为label predictor的loss - lamb * domain classifier的loss
        # 因为feature extractor希望骗过domain classifier，要最大化其loss，想去相当于梯度上升
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        # FeatureExtrator 和 LabelPredictor 的loss
        running_F_loss+= loss.item()
        # 反向传播
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        # 计算source data中label predictor预测正确的数量
        correct += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]

    # 返回domain classifier的平均loss、feature extracor和lable predictor的平均loss、source data分类正确率
    return running_D_loss / (i+1), running_F_loss / (i+1), correct / total_num


if __name__ == '__main__':
    train_path = sys.argv[1]
    # path+'train_data/'
    test_path = sys.argv[2]
    # path+'testdata_raw/'
    extractor_dir = sys.argv[3]
    # extractor_model.param
    predictor_dir = sys.argv[4]
    # predicotr_model.param


    # source data的变换
    source_transform = transforms.Compose([
        # 变成灰度图传入cv2.canny
        transforms.Grayscale(),
        # cv2.Cannay需要np.array传入，但是DataLoader传进来的不是np.array，需要转换
        transforms.Lambda(lambda x: cv2.Canny(np.array(x), 150, 300)),
        # 将np.array转换为PILImage进行后面的转换
        transforms.ToPILImage(),
        # data统一转换成64*64，方便之后搭模型
        transforms.Resize(64),
        # 数据增强
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    # target data的变换
    target_transform = transforms.Compose([
        # 和target data一样统一变成灰度图
        transforms.Grayscale(),
        # 转换成64*64
        transforms.Resize(64),
        # 数据增强
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    # 直接用ImageFolder读取source_dataset和target_dataset
    source_dataset = ImageFolder(train_path, transform=source_transform)
    target_dataset = ImageFolder(test_path, transform=target_transform)

    # load data
    source_dataloader = DataLoader(source_dataset, batch_size=64, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=64, shuffle=True)


    # 初始化模型
    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor().cuda()
    domain_classifier = DomainClassifier().cuda()

    # label predictor用较差熵损失函数
    class_criterion = nn.CrossEntropyLoss()
    # domain classifier判断是source data还是target data，二分类，用sigmoid+BCE Loss
    domain_criterion = nn.BCEWithLogitsLoss()

    # 用Adam优化
    optimizer_F = optim.Adam(feature_extractor.parameters())
    optimizer_C = optim.Adam(label_predictor.parameters())
    optimizer_D = optim.Adam(domain_classifier.parameters())


    # 训练150个epoch
    for epoch in range(150):
        train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=0.1)
        print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch+1, train_D_loss, train_F_loss, train_acc))

    # 保存模型
    torch.save(feature_extractor.state_dict(), extractor_dir)
    torch.save(label_predictor.state_dict(), predictor_dir)