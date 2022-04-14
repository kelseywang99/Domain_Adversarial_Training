

# README


## train.sh

python3 train.py $1 $2 extractor_model.param predictor_model.param

# $1: 训练数据的文件夹：path/train_data/ 

# $2: 测试数据的文件夹：path/testdat_raw/

# 要保存extractor、predictor的参数

需要说明的用到的包：
- PIL.Image、cv2：用于读入图片


## test.sh 

test.py $1 extractor_model.param predictor_model.param $2

$1: 测试数据的文件夹：path/testdata_raw/ 

$2: 要保存的生成文件：path/res.csv

需要说明的用到的包：
- PIL.Image：用于读入图片



## compare_and_draw.py
用于视觉化并对比真实图片以及手绘图片分别通过没有使用、使用 domain adversarial training 的 feature extractor 的 domain 分布图