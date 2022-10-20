简体中文 | [English](../en/benchmark.md)

# Benchmark


本文档给出了PaddleVideo系列模型在各平台预测耗时benchmark。

---

## 目录

- [1. 视频分类模型](#1)
    - [1.1 测试数据](#11)
    - [1.2 测试环境](#12)
    - [1.3 测试结果](#13)
        - [1.3.1 GPU推理速度一览](#131)
        - [1.3.2 CPU推理速度一览](#132)
    - [1.4 测试方法](#14)
        - [1.4.1 单个模型测试](#141)
        - [1.4.2 批量测试](#141)
- [2. 时序分割模型](#2)
    - [2.1 测试环境](#21)
    - [2.2 测试结果](#2)


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

### 1.2 测试环境

硬件环境：

- CPU: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
- GPU: Tesla V100 16G

软件环境：
- Python 3.7
- PaddlePaddle 2.3.1
- CUDA 10.2
- CUDNN 8.1.1
- 各python库版本参考[requirement.txt](../../requirements.txt)

### 1.3 测试结果

#### 1.3.1 GPU推理速度一览

各模型性能数据按预测总时间排序，结果如下:

|模型名称 | 骨干网络 | 配置文件 | 精度% | 预处理时间ms | 模型推理时间ms | 预测总时间ms |
| :---- | :---- | :----: |:----: |:----: |:----: |:----: |
| PP-TSM | MobileNetV2 | [pptsm_mv2_k400_videos_uniform.yaml](../../configs/recognition/pptsm/pptsm_mv2_k400_videos_uniform.yaml) | 68.09 | 51.5 | 3.31 | 54.81 |
| PP-TSM | MobileNetV3 | [pptsm_mv3_k400_frames_uniform.yaml](../../configs/recognition/pptsm/pptsm_mv3_k400_frames_uniform.yaml) | 69.84 | 51 | 4.34 | 55.34 |
| **PP-TSMv2** | PP-LCNet_v2.8f |	[pptsm_lcnet_k400_8frames_uniform.yaml](../../configs/recognition/pptsm/v2/pptsm_lcnet_k400_8frames_uniform.yaml) | **72.45**| 55.31 | 4.37 | **59.68** |
| TSM | R50 | [tsm_k400_frames.yaml](../../configs/recognition/tsm/tsm_k400_frames.yaml) | 71.06 | 52.02 | 9.87 | 61.89 |
|**PP-TSM**	| R50 |	[pptsm_k400_frames_uniform.yaml](../../configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml) | **75.11** | 51.84 | 11.26 | **63.1** |
|PP-TSM	| R101 | [pptsm_k400_frames_dense_r101.yaml](../../configs/recognition/pptsm/pptsm_k400_frames_dense_r101.yaml) | 76.35| 52.1 | 17.91 | 70.01 |
| PP-TSMv2 | PP-LCNet_v2.16f |	[pptsm_lcnet_k400_16frames_uniform.yaml](../../configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform.yaml) | 74.38 |  69.4 | 7.55 | 76.95 |
| SlowFast | 4*16 |	[slowfast.yaml](../../configs/recognition/slowfast/slowfast.yaml) | 74.35 | 99.27 | 27.4 | 126.67 |
| *VideoSwin | B | [videoswin_k400_videos.yaml](../../configs/recognition/videoswin/videoswin_k400_videos.yaml) | 82.4 | 95.65 | 117.22 | 212.88 |
| MoViNet | A0 | [movinet_k400_frame.yaml](../../configs/recognition/movinet/movinet_k400_frame.yaml) | 66.62 | 150.36 | 47.24 | 197.60 |
| *PP-TimeSformer | base | [pptimesformer_k400_videos.yaml](../../configs/recognition/pptimesformer/pptimesformer_k400_videos.yaml) | 78.87 | 299.48 | 133.41 | 432.90 |
| *TimeSformer |	base |	[timesformer_k400_videos.yaml](../../configs/recognition/timesformer/timesformer_k400_videos.yaml) | 77.29 | 301.54 | 136.12 | 437.67 |
| TSN | R50	| [tsn_k400_frames.yaml](../../configs/recognition/tsn/tsn_k400_frames.yaml) | 69.81 | 794.30 | 168.70 | 963.00 |
| PP-TSN | R50 | [pptsn_k400_frames.yaml](../../configs/recognition/pptsn/pptsn_k400_frames.yaml) | 75.06 | 837.75 | 175.12 | 1012.87 |

* 注:带`*`表示该模型未使用tensorRT进行预测加速。

- TSN预测时采用TenCrop，比TSM采用的CenterCrop更加耗时。TSN如果使用CenterCrop，则速度稍优于TSM，但精度会低3.5个点。

#### 1.3.2 CPU推理速度一览

各模型性能数据按预测总时间排序，结果如下:

|模型名称 | 骨干网络 | 配置文件 | 精度% | 预处理时间ms | 模型推理时间ms | 预测总时间ms |
| :---- | :---- | :----: |:----: |:----: |:----: |:----: |
| PP-TSM | MobileNetV2 | [pptsm_mv2_k400_videos_uniform.yaml](../../configs/recognition/pptsm/pptsm_mv2_k400_videos_uniform.yaml) | 68.09 | 52.62 | 137.03 | 189.65 |
| PP-TSM | MobileNetV3 | [pptsm_mv3_k400_frames_uniform.yaml](../../configs/recognition/pptsm/pptsm_mv3_k400_frames_uniform.yaml) | 69.84| 53.44 | 139.13 | 192.58 |
| **PP-TSMv2** | PP-LCNet_v2.8f |	[pptsm_lcnet_k400_8frames_uniform.yaml](../../configs/recognition/pptsm/v2/pptsm_lcnet_k400_8frames_uniform.yaml) | **72.45**| 53.37 | 189.62 | **242.99** |
| **PP-TSMv2** | PP-LCNet_v2.16f |	[pptsm_lcnet_k400_16frames_uniform.yaml](../../configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform.yaml) | **74.38**|  68.07 | 388.64 | **456.71** |
| SlowFast | 4*16 |	[slowfast.yaml](../../configs/recognition/slowfast/slowfast.yaml) | 74.35 | 110.04 | 1201.36 | 1311.41 |
| TSM | R50 | [tsm_k400_frames.yaml](../../configs/recognition/tsm/tsm_k400_frames.yaml) | 71.06 | 52.47 | 1302.49 | 1354.96 |
|PP-TSM	| R50 |	[pptsm_k400_frames_uniform.yaml](../../configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml) | 75.11 | 52.26  | 1354.21 | 1406.48 |
|*MoViNet | A0 | [movinet_k400_frame.yaml](../../configs/recognition/movinet/movinet_k400_frame.yaml) | 66.62 | 148.30 |	1290.46 | 1438.76 |
|PP-TSM	| R101 | [pptsm_k400_frames_dense_r101.yaml](../../configs/recognition/pptsm/pptsm_k400_frames_dense_r101.yaml) | 76.35| 52.50 | 2236.94 | 2289.45 |
| PP-TimeSformer | base | [pptimesformer_k400_videos.yaml](../../configs/recognition/pptimesformer/pptimesformer_k400_videos.yaml) | 78.87 | 294.89	| 13426.53 | 13721.43 |
| TimeSformer |	base |	[timesformer_k400_videos.yaml](../../configs/recognition/timesformer/timesformer_k400_videos.yaml) | 77.29 | 297.33 |	14034.77 |	14332.11 |
| TSN | R50	| [tsn_k400_frames.yaml](../../configs/recognition/tsn/tsn_k400_frames.yaml) | 69.81 | 860.41 | 18359.26 | 19219.68 |
| PP-TSN | R50 | [pptsn_k400_frames.yaml](../../configs/recognition/pptsn/pptsn_k400_frames.yaml) | 75.06 | 835.86 | 19778.60 | 20614.46 |
| *VideoSwin | B | [videoswin_k400_videos.yaml](../../configs/recognition/videoswin/videoswin_k400_videos.yaml) | 82.4 | 76.21 | 32983.49 | 33059.70 |


* 注: 带`*`表示该模型未使用mkldnn进行预测加速。


### 1.4 测试方法

在进行测试之前，需要安装[requirements.txt](../../requirements.txt)相关依赖，并且还需安装`AutoLog`用于记录计算时间，使用如下命令安装:
```bash
python3.7 -m pip install --upgrade pip
pip3.7 install --upgrade -r requirements.txt
python3.7 -m pip install git+https://github.com/LDOUBLEV/AutoLog
```

#### 1.4.1 单个模型测试

以PP-TSM模型为例，请先参考[PP-TSM文档](./model_zoo/recognition/pp-tsm.md)导出推理模型，之后使用如下命令进行速度测试：

```python
python3.7 tools/predict.py --input_file time-test/file.list \
                          --time_test_file=True \
                          --config configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml \
                          --model_file inference/ppTSM/ppTSM.pdmodel \
                          --params_file inference/ppTSM/ppTSM.pdiparams \
                          --use_gpu=False \
                          --use_tensorrt=False \
                          --enable_mkldnn=True \
                          --enable_benchmark=True \
                          --disable_glog True
```

- 各参数含义如下：

```txt
input_file:     指定测试文件/文件列表, 示例使用1.1小节提供的测试数据
time_test_file: 是否进行时间测试，请设为True
config:         指定模型配置文件
model_file:     指定推理文件pdmodel路径
params_file:    指定推理文件pdiparams路径
use_gpu:        是否使用GPU预测, False则使用CPU预测
use_tensorrt:   是否开启TensorRT预测
enable_mkldnn:  开启benchmark时间测试，默认设为True
disable_glog:   是否关闭推理时的日志，请设为True
```

- 测试时，GPU推理使用FP32+TensorRT配置下，CPU使用mkldnn加速。运行100次，去除前3次的warmup时间，得到推理平均时间。

#### 1.4.2 批量测试

使用以下批量测试脚本，可以方便的将性能结果进行复现：

- 1. 下载预训练模型:

```bash
mkdir ckpt
cd ckpt
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_uniform_distill.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSM_k400_uniform_distill_r101.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_mv2_k400.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_mv3_k400.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.3/PPTSMv2_k400_16f_dml.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSN_k400_8.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTimeSformer_k400_8f_distill.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.1/TSM/TSM_k400.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TSN_k400.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TimeSformer_k400.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.3/MoViNetA0_k400.pdparams
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/VideoSwin_k400.pdparams
```

- 2. 准备各模型配置参数列表文件`model.list`

```txt
PP-TSM_R50      configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml        ckpt/ppTSM_k400_uniform_distill.pdparams ppTSM
PP-TSM_R101     configs/recognition/pptsm/pptsm_k400_frames_dense_r101.yaml     ckpt/ppTSM_k400_uniform_distill_r101.pdparams ppTSM
PP-TSM_MobileNetV2      configs/recognition/pptsm/pptsm_mv2_k400_videos_uniform.yaml    ckpt/ppTSM_mv2_k400.pdparams ppTSM
PP-TSM_MobileNetV3      configs/recognition/pptsm/pptsm_mv3_k400_frames_uniform.yaml    ckpt/ppTSM_mv3_k400.pdparams ppTSM
PP-TSMv2_PP-LCNet_v2    configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform_dml_distillation.yaml      ckpt/PPTSMv2_k400_16f_dml.pdparams ppTSMv2
PP-TSN_R50      configs/recognition/pptsn/pptsn_k400_frames.yaml        ckpt/ppTSN_k400_8.pdparams ppTSN
PP-TimeSformer_base     configs/recognition/pptimesformer/pptimesformer_k400_videos.yaml        ckpt/ppTimeSformer_k400_8f_distill.pdparams ppTimeSformer
TSM_R50 configs/recognition/tsm/tsm_k400_frames.yaml    ckpt/TSM_k400.pdparams TSM
TSN_R50 configs/recognition/tsn/tsn_k400_frames.yaml    ckpt/TSN_k400.pdparams TSN
TimeSformer_base        configs/recognition/timesformer/timesformer_k400_videos.yaml    ckpt/TimeSformer_k400.pdparams TimeSformer
SlowFast_416    configs/recognition/slowfast/slowfast.yaml      ckpt/SlowFast.pdparams SlowFast
MoViNet_A0      configs/recognition/movinet/movinet_k400_frame.yaml     ckpt/MoViNetA0_k400.pdparams MoViNet
VideoSwin_B     configs/recognition/videoswin/videoswin_k400_videos.yaml        ckpt/VideoSwin_k400.pdparams VideoSwin
```

- 3. 批量导出模型，执行时传入model.list文件

```bash
file=$1

while read line
do
    arr=($line)
    ModelName=${arr[0]}
    ConfigFile=${arr[1]}
    ParamsPath=${arr[2]}
    echo $ModelName

    python3.7 tools/export_model.py -c $ConfigFile \
                                    -p $ParamsPath \
                                    -o inference/$ModelName
done <$file
```

- 4. 测试时间，执行时传入model.list文件

```bash
file=$1

while read line
do
    arr=($line)
    ModelName=${arr[0]}
    ConfigFile=${arr[1]}
    ParamsPath=${arr[2]}
    Model=${arr[3]}

    python3.7 tools/predict.py --input_file ../../time-test/file.list \
                            --time_test_file=True \
                            --config $ConfigFile \
                            --model_file inference/$ModelName/$Model.pdmodel \
                            --params_file inference/$ModelName/$Model.pdiparams \
                            --use_gpu=False \
                            --use_tensorrt=False \
                            --enable_mkldnn=False \
                            --enable_benchmark=True \
                            --disable_glog True
    echo =====$ModelName END====
done <$file
```

---

## 2. 时序分割模型

### 2.1 测试环境

硬件环境:

- 8 NVIDIA Tesla V100 (16G) GPUs
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

软件环境:

- Python 3.7
- PaddlePaddle2.0
- CUDA 10.1
- CUDNN 7.6.3
- NCCL 2.1.15
- GCC 8.2.0

### 2.2 测试结果

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
