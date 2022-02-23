简体中文 | [English](../en/benchmark.md)

# Benchmark

此文档主要对比PaddleVideo模型库与主流模型库的训练速度。


## 环境配置
### 硬件环境

- 8 NVIDIA Tesla V100 (16G) GPUs
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

### 软件环境

- Python 3.7
- PaddlePaddle2.0
- CUDA 10.1
- CUDNN 7.6.3
- NCCL 2.1.15
- GCC 8.2.0


### 实验与评价指标

实验中我们测量了平均训练时间，包括数据处理和模型训练两个部分，训练速度均采用每秒钟训练的样本数量(ips)来计量,
数值越大说明训练速度越快，并且考虑到机器预热的问题，前50次迭代的时间没有被计算在内。

在相同的数据和模型配置下对比了PaddleVideo和其他的视频理解工具箱，为了保证比较的公平性，对比实验都是在相同的硬件条件下进行，实验所用数据请参考[数据准备](./dataset/k400.md)
观察下表可以发现PaddleVideo相比其他的视频理解框架在训练速度方面取得了巨大的提升，尤其是[Slowfast](../../configs/recognition/slowfast/slowfast.yaml)模型取得了将近一倍的训练速度的提升。
对于每一种模型配置，我们采用了相同的数据预处理方法并且保证输入是相同的。

## 实验结果
### 分类模型

| Model | batch size <sub>x</sub> gpus | PaddleVideo(ips) | Reference(ips) | MMAction2 (ips)  | PySlowFast (ips)|
| :------: | :-------------------:|:---------------:|:---------------: | :---------------:  |:---------------: |
| [TSM](../../configs/recognition/tsm/tsm.yaml) | 16x8 | 58.1 | 46.04(temporal-shift-module) | To do | X |
| [PPTSM](../../configs/recognition/tsm/pptsm.yaml) | 16x8 |  57.6 | X |    X   | X |
| [TSN](../../configs/recognition/tsn/tsn.yaml) | 16x8 |  To do |  To do (tsn-pytorch) | To do | X |
| [Slowfast](../../configs/recognition/slowfast/slowfast.yaml)| 16x8 | 99.5 | X | To do | 43.2 |
| [Attention_LSTM](../../configs/recognition/attention_lstm/attention_lstm.yaml) |  128x8  | 112.6  | X | X | X |


### 定位模型

| Model | PaddleVideo(ips) |MMAction2 (ips) |BMN(boundary matching network) (ips)|
| :--- | :---------------: | :-------------------------------------: | :-------------------------------------: |
| [BMN](../../configs/localization/bmn.yaml)  | 43.84 | x | x |

### 分割模型

本仓库提供经典和热门时序动作分割模型的性能和精度对比

| Model | Metrics | Value | Flops(M) |Params(M) | test time(ms) bs=1 | test time(ms) bs=2 | inference time(ms) bs=1 | inference time(ms) bs=2 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MS-TCN | F1@0.5 | 38.8% | 791.360 | 0.8 | 170 | - | 10.68 | - |
| ASRF | F1@0.5 | 55.7% | 1,283.328 | 1.3 | 190 | - | 16.34 | - |

* 模型名称：填写模型的具体名字，比如PP-TSM
* Metrics：填写模型测试时所用的指标，使用的数据集为**breakfast**
* Value：填写Metrics指标对应的数值，一般保留小数点后两位
* Flops(M)：模型一次前向运算所需的浮点运算量，可以调用PaddleVideo/tools/summary.py脚本计算（不同模型可能需要稍作修改），保留小数点后一位，使用数据**输入形状为(1, 2048, 1000)的张量**测得
* Params(M)：模型参数量，和Flops一起会被脚本计算出来，保留小数点后一位
* test time(ms) bs=1：python脚本开batchsize=1测试时，一个样本所需的耗时，保留小数点后两位。测试使用的数据集为**breakfast**。
* test time(ms) bs=2：python脚本开batchsize=2测试时，一个样本所需的耗时，保留小数点后两位。时序动作分割模型一般是全卷积网络，所以训练、测试和推理的batch_size都是1。测试使用的数据集为**breakfast**。
* inference time(ms) bs=1：推理模型用GPU（默认V100）开batchsize=1测试时，一个样本所需的耗时，保留小数点后两位。推理使用的数据集为**breakfast**。
* inference time(ms) bs=2：推理模型用GPU（默认V100）开batchsize=1测试时，一个样本所需的耗时，保留小数点后两位。时序动作分割模型一般是全卷积网络，所以训练、测试和推理的batch_size都是1。推理使用的数据集为**breakfast**。
