## 实验结果记录

模型推断执行：`train.py`中的`inference`

|experiment index|num_classes|base_n_filter|init lr|epochs|crop size|aug|init weights|train output|inference result|train mode|
|-|-|-|-|-|-|-|-|-|-|-|
|train.sh/exp1|2|6|2e-4|400|384 384 256|seg_train|common_seg_epoch_28_train_0.069|common_seg_epoch_46_train_0.060||model.eval()|

### 实验结果记录1

包含在训练集中的数据
![](./img/exp1/1.2.392.200036.9116.2.2054276706.1582589798.9.1347400003.1.gif)

测试集中的数据
![](./img/exp1/1.2.392.200036.9116.2.2054276706.1589264256.12.1245900005.1.gif)
