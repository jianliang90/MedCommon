## 数据集

训练数据路径:`/data/medical/cardiac/chamber/seg/chamber_256`
训练过程中数据划分配置：`/data/medical/cardiac/chamber/seg/chamber_256/config`

生成过程:`./datasets/data_preprocessing.py`

## 训练

模型推断执行：`train.py`中的`inference`

|experiment index|num_classes|base_n_filter|init lr|epochs|crop size|aug|init weights|train output|inference result|train mode|auto resize|
|-|-|-|-|-|-|-|-|-|-|-|-|
|train.sh/exp1|2|6|2e-4|400|256 256 256|seg_train|None|common_seg_epoch_19_train_0.067||model.eval()|True|

### 实验结果记录1

包含在训练集中的数据
![](./img/exp1/1.3.12.2.1107.5.1.4.60320.30000015020202382801500025712.gif)

测试集中的数据
![](./img/exp1/1.3.12.2.1107.5.1.4.60320.30000018042800095349200029782.gif)