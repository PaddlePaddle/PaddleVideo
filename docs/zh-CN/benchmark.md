简体中文 | [English](../en/benchmark.md)

# Benchmark

此文档主要介绍PaddleVideo模型训练速度上与主流模型库间的横向对比。

## 环境配置
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

### 统计试验

实验中我们测量了平均训练时间，包括数据处理和模型训练两个部分，训练速度均采用每秒钟训练的样本数量(ips)来计量,
数值越大说明训练速度越快，并且考虑到机器预热的问题，前50次迭代的时间没有被计算在内。

在相同的数据和模型配置下对比了PaddleVideo和其他的视频理解工具箱，为了保证比较的公平性，对比实验都是在相同的硬件条件下进行。
观察下表可以发现PaddleVideo相比其他的视频理解框架在训练速度方面取得了巨大的提升，尤其是**Slowfast**模型取得了将近一倍的训练速度的提升。
对于每一种模型配置，我们采用了相同的数据预处理方法并且保证输入是相同的。

## 结论
### 视频分类

|| Model | batch size <sub>x</sub> gpus | PaddleVideo(ips) | Reference(ips) | MMAction2 (ips)  | PySlowFast (ips)|
| :------: | :-------------------:|:---------------:|:---------------: | :---------------:  |:---------------: |
| [TSM](model_zoo/recognition/tsm.md) | 16x8 | 58.1 | 46.04(temporal-shift-module) | <sup>ToDo</sup> | X |
| [PPTSM](model_zoo/recognition/pp-tsm.md) | 16x8 |  57.6 | X |    X   | X |
| [TSN](model_zoo/recognition/tsn.md) | 16x8 |  841.1 |  <sup>ToDo (tsn-pytorch)</sup> | <sup>ToDo</sup> | X | 
| [Slowfast](model_zoo/recognition/slowfast.md)| 16x8 | 99.5 | X | <sup>ToDo</sup> | 43.2 |
| [Attention_LSTM](model_zoo/recognition/attention_lstm.md) |  128x8  | 112.6  | X | X | X |

### 动作定位

| Model | PaddleVideo(ips) |MMAction2 (ips) |BMN(boundary matching network) (ips)|
| :--- | :---------------: | :-------------------------------------: | :-------------------------------------: |
| [BMN](model_zoo/localization/bmn.md)  | <sup>ToDo</sip> | x | x |
