#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

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

# 为了展现domain adversarial training在训练之后，
# 能够让source data和target data经过feature extracotr的特征分布类似，
# 对比source data和target data在经过不采用domain adversarial training方法得到的feature extractor、
# 和采用domain adversarial training方法得到的feature extractor后，得到的feature分布图

# 训练一个不用domain adversarial training的feature extractor和label predictor，
# 网络结构和采用domain adversarial training的feature extractor和label predictor相同
def train_epoch_no_dann(source_dataloader):
    # running_D_loss: DomainClassifier的loss
    # running_F_loss: FeatureExtrator 和 LabelPredictor 的loss
    running_D_loss, running_F_loss = 0.0, 0.0
    # correct: 分类正确的个数 
    # total_num: source data总共的个数
    correct, total_num = 0.0, 0.0

    i = 0
    for (source_data, source_label) in source_dataloader:
        # 提取data
        source_data = source_data.cuda()
        source_label = source_label.cuda()

        feature = feature_extractor(source_data)
        # source data经过feature extractor后的feature作为input，用label predictor预测label
        class_logits = label_predictor(feature)
        loss = class_criterion(class_logits, source_label)
        # FeatureExtrator 和 LabelPredictor 的loss
        running_F_loss+= loss.item()
        # 反向传播
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        # 计算source data中label predictor预测正确的数量
        correct += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        i += 1

    # 返回feature extracor和lable predictor的平均loss、source data分类正确率
    return running_F_loss / (i+1), correct / total_num

if __name__ == '__main__':
    train_path = sys.argv[1]
    # path/train_data/ 
    test_path = sys.argv[2]
    # path/testdata_raw/
    dann_extractor_param = sys.argv[3]
    # 用dann方法训练得到的extractor参数：path/extractor_model.param

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

    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor().cuda()

    optimizer_F = optim.Adam(feature_extractor.parameters())
    optimizer_C = optim.Adam(label_predictor.parameters())

    class_criterion = nn.CrossEntropyLoss()

    # ### 1、不采用domain adversarial training的feature分布
    # 
    # 不用dann：训练150 epochs
    for epoch in range(150):
        train_F_loss, train_acc = train_epoch_no_dann(source_dataloader)
        print('epoch {:>3d}: train F loss: {:6.4f}, acc {:6.4f}'.format(epoch+1, train_F_loss, train_acc))

    # 然后，对上述不用domain adversarial training得到的模型，
    # 看一下source data和target data经过feature extractor得到的feature的分布。先来得到feature。
    feature_extractor.eval()

    source_dataloader = DataLoader(source_dataset, batch_size=64, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=64, shuffle=True)

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.cuda()
        target_data = target_data.cuda()

        # 把source data和target data拼接在一起
        mixed_data = torch.cat([source_data, target_data], dim=0)
        # 设置domain_label，source是1，target是0
        domain_label_temp = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        domain_label_temp[:source_data.shape[0]] = 1
        # 第一次进入循环，建立feature、domain_label直接得到
        if i == 0: 
            feature = feature_extractor(mixed_data)
            domain_label = domain_label_temp.clone()
        # 非第一次进入循环，feature、domain_label与已经得到的feature、domain_label拼接在一起
        else:
            feature = torch.cat([feature, feature_extractor(mixed_data)], dim=0)
            domain_label =  torch.cat([domain_label, domain_label_temp], dim=0)
        # 为了节省时间，做16个batch，大概用了source和target各用了1000个的数据，足以说明问题
        if i == 15:
            break

    # 之后，进行数据降维和可视化。
    # 根据网络结构，feature extractor输出的feature是2048维，
    # 为了可视化，采用TSNE方法降维到2维。
    # 可以明显看出，source data和target data得到的feature分布差异很大。

    # 将feature进行数据降维，从2048降到2，方便在平面图上进行展示
    from sklearn import (manifold, datasets)
    X=torch.Tensor.cpu((feature.reshape(feature.shape[0], 512*2*2)) / 255.0).detach()
    y=torch.Tensor.cpu(domain_label).detach().numpy()
    # 采用TSNE降维的方法，首先运用PCA的方法，然后映射到2维
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    # 对feature进行降维
    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    
    # 降维后进行可视化
    # 归一化
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  
    # 画图
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(int(y[i])))
    plt.xticks([])
    plt.yticks([])
    plt.show()


    # ### 2、采用domain adversarial training的feature分布
    # 
    # 对之前采用domain adversarial training得到的模型，
    # 看一下source data和target data经过feature extractor得到的feature的分布。
    # 先来得到feature。

    feature_extractor = FeatureExtractor().cuda()
    feature_extractor.load_state_dict(torch.load(dann_extractor_param))
    feature_extractor.eval()

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.cuda()
        target_data = target_data.cuda()

        # 把source data和target data拼接在一起
        mixed_data = torch.cat([source_data, target_data], dim=0)
        # 设置domain_label，source是1，target是0
        domain_label_temp = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        domain_label_temp[:source_data.shape[0]] = 1
        # 第一次进入循环，建立feature、domain_label直接得到
        if i == 0: 
            feature = feature_extractor(mixed_data)
            domain_label = domain_label_temp.clone()
        # 非第一次进入循环，feature、domain_label与已经得到的feature、domain_label拼接在一起
        else:
            feature = torch.cat([feature, feature_extractor(mixed_data)], dim=0)
            domain_label =  torch.cat([domain_label, domain_label_temp], dim=0)
        # 为了节省时间，做16个batch，大概用了source和target各用了1000个的数据，足以说明问题
        if i == 15:
            break

    # 之后，进行数据降维和可视化。 \
    # 根据网络结构，feature extractor输出的feature是2048维，
    # 为了可视化，采用TSNE方法降维到2维。可以看出，source data和target data得到的feature分布几乎没有差异。


    X=torch.Tensor.cpu((feature.reshape(feature.shape[0], 512*2*2)) / 255.0).detach()
    y=torch.Tensor.cpu(domain_label).detach().numpy()
    # 采用TSNE降维的方法
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    # 对feature进行降维
    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    
    # 降维后进行可视化
    # 归一化
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  
    # 画图
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(int(y[i])))
    plt.xticks([])
    plt.yticks([])
    plt.show()

