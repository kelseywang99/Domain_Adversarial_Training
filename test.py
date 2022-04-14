#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

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


if __name__ == '__main__':
    test_path = sys.argv[1]
    # testdata_raw/ 
    e_param = sys.argv[2]
    # extractor_model.param
    p_param = sys.argv[3]
    # predictor_model.param
    save_path = sys.argv[4]
    # res.csv

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

     # 读取dataset
    target_dataset = ImageFolder(test_path, transform=target_transform)
    test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

    # 初始化模型
    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor().cuda()

    feature_extractor.load_state_dict(torch.load(e_param))
    label_predictor.load_state_dict(torch.load(p_param))

    # 用来储存预测结果
    result = []
    label_predictor.eval()
    feature_extractor.eval()

    for i, (test_data, _) in enumerate(test_dataloader):
        test_data = test_data.cuda()
        # 用feature extractor + label predictor预测
        class_logits = label_predictor(feature_extractor(test_data))
        # 得到预测结果
        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        # 加入到列表中
        result.append(x)

    # 写入CSV保存
    result = np.concatenate(result)
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv(save_path,index=False)
