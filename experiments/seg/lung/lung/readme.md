## 数据集

训练数据路径:`/data/medical/data/lung/LUNA/lung_256`
训练过程中数据划分配置：`/data/medical/data/lung/LUNA/lung_256/config`

生成过程:`./train.py`，具体见文件中的注释

## 训练

## 实验结果记录

模型推断执行：`train.py`中的`inference`

|experiment index|num_classes|base_n_filter|init lr|epochs|crop size|aug|init weights|train output|inference result|train mode|auto resize|
|-|-|-|-|-|-|-|-|-|-|-|-|
|train.sh/exp1|2|6|2e-4|400|256 256 256|seg_train|None|common_seg_epoch_91_train_0.024||model.eval()|True|

### 实验结果记录1

包含在训练集中的数据
![](./img/exp1/1.3.6.1.4.1.14519.5.2.1.6279.6001.640729228179368154416184318668.gif)

测试集中的数据
![](./img/exp1/1.3.6.1.4.1.14519.5.2.1.6279.6001.608029415915051219877530734559.gif)