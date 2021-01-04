# Benchmark

此文档主要介绍Paddle版本的模型与竞品版本模型以及官方版本模型的训练速度对比实验结果。

## 实验环境配置介绍
### 硬件环境

- 8 NVIDIA Tesla V100 (16G) GPUs
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

### 软件环境

- Python 3.7
- Paddlepaddle-develop
- CUDA 10.1
- CUDNN 7.6.3
- NCCL 2.1.15
- gcc 8.2.0

### 评价指标

实验中我们将测量平均训练时间，包括数据处理和模型训练两个部分，训练速度均采用每秒钟训练的样本数量(ips)来计量,
数值越大说明训练速度越快，并且考虑到机器预热的问题，前50次迭代的时间没有被计算在内

### 比较规则和方法

我们在相同的数据和模型配置下对比Paddle视频库和其他的视频理解工具箱，为了保证比较的公平性，对比实验都是在相同的硬件条件下进行。观察下表可以发现Paddle视频库相比其他的视频理解框架在
训练速度方面取得了巨大的提升，尤其是Slowfast模型取得了将近一倍的训练速度的提升。对于每一种模型配置，我们采用了相同的数据预处理方法并且保证输入是相同的。

## Main Results
### Recognizers

| Model | batch size x gpus | Paddle(ips) | Reference(ips) | MMAction2 (ips)  | PySlowFast (ips)|
| :------ :| :-------------------:|:---------------:| :---------------: | :---------------:  |:---------------:  |
| [TSM](../configs/recognition/tsm/tsm.yaml) |     16x8         |  58.1 |  46.04(temporal-shift-module) |  To do | X |
| [PPTSM](../configs/recognition/tsm/pptsm.yaml) |   16x8         |  57.6 |   X    |    X   | X |
| [TSN](../configs/recognition/tsn/tsn.yaml) |     16x8         |  841.1 |  To do (tsn-pytorch) |  To do | X | 
| [Slowfast](../configs/recogntion/slowfast/slowfast.yaml)|  16x8         |  99.5  |   X    |  To do | 43.2 |
| [Attention_LSTM](../configs/recognition/attention_lstm/attention_lstm.yaml) |  128x8  | 112.8  |   X    |   X    |   X  |

### Localizers

| Model | Paddle(ips) |MMAction2 (ips) |BMN(boundary matching network) (ips)|
| :--- | :---------------: | :-------------------------------------: | :-------------------------------------: |
| [BMN](../configs/localization/bmn.yaml)  | To do | x | x |
