#!/usr/bin/env bash

python3 train.py $1 $2 extractor_model.param predictor_model.param

# $1: 训练数据的文件夹：path/train_data/ 
# $2: 测试数据的文件夹：path/testdat_raw/
# 要保存extractor、predictor的参数