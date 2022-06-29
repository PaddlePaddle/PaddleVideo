简体中文 | [English](../en/benchmark.md)

# Benchmark

本文档给出了PaddleVideo系列模型在各平台预测耗时benchmark。

## 1. 视频分类模型

### 1.1 测试数据

我们从Kinetics-400数据集中，随机选择提供100条用于benchmark时间测试，测试数据可以[点击](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/time-test.tar)下载。

解压后文件目录：
```txt
time-test
├── data       # 测试视频文件
└── file.list  # 文件列表
```

视频属性如下:

```txt
mean video time:  9.67s
mean video width:  373
mean video height:  256
mean fps:  25
```

### 1.2 测试方法

#### 1.2.1 测试环境

- 硬件环境

- 软件环境

#### 1.2.2 测试脚本

以PP-TSM为例，请先参考模型文档[PP-TSM]()下载推理模型，之后使用如下命令进行速度测试：

```python
python3.7 tools/predict.py --input_file time-test/file.list \
                          --config configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml \
                          --model_file inference/ppTSM/ppTSM.pdmodel \
                          --params_file inference/ppTSM/ppTSM.pdiparams \
                          --use_gpu=False \
                          --use_tensorrt=False \
                          --enable_mkldnn=True \
                          --enable_benchmark=True
```

各参数含义如下：

```txt
enable_mkldnn: 开启benchmark时间测试，请设为True
input_file:    指定测试文件/文件列表, 示例使用1.1小节提供的测试数据
config:        指定模型配置文件
model_file:    指定推理文件pdmodel路径
params_file:   指定推理文件pdiparams路径
use_gpu:       是否使用GPU预测, False则使用CPU预测
use_tensorrt:  是否开启TensorRT预测
```

### 1.3 测试结果

- GPU推理速度一览
- CPU推理速度一览

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
