#!/usr/bin/env bash

python3 test.py $1 extractor_model.param predictor_model.param $2

# $1: 测试数据的文件夹：path/testdata_raw/ 
# $2: 要保存的生成文件：path/res.csv