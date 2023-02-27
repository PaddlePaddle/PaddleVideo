[English](../../../en/model_zoo/recognition/pp-tsm.md) | 简体中文

# PP-TSM视频分类模型

---
## 目录

- [1. 简介](#1)
- [2. 性能benchmark](#2)
- [3. 数据准备](#3)
- [4. 模型训练](#4)
    - [4.1 预训练模型下载](#41)
    - [4.2 多卡训练](#42)
    - [4.3 蒸馏训练](#43)
    - [4.4 配置文件说明](#44)
    - [4.5 配置文件推荐使用](#45)
- [5. 模型测试](#5)
    - [5.1 中心采样测试](#51)
    - [5.2 密集采样测试](#52)
- [6. 模型推理部署](#6)
    - [6.1 导出推理模型](#61)
    - [6.2 基于python预测引擎推理](#62)
    - [6.3 基于c++预测引擎推理](#63)
    - [6.4 服务化部署](#64)
    - [6.5 Paddle2ONNX 模型预测与转换](#65)
- [7. 模型库下载](#7)
- [8. 参考论文](#8)

<a name="1"></a>
## 1. 简介

视频分类与图像分类相似，均属于识别任务，对于给定的输入视频，视频分类模型需要输出其预测的标签类别。如果标签都是行为类别，则该任务也常被称为**行为识别**。与图像分类不同的是，视频分类往往需要利用多帧图像之间的时序信息。PP-TSM是PaddleVideo自研的实用产业级视频分类模型，在实现前沿算法的基础上，考虑精度和速度的平衡，进行模型瘦身和精度优化，使其可能满足产业落地需求。

### PP-TSM

PP-TSM基于ResNet-50骨干网络进行优化，从数据增强、网络结构微调、训练策略、BN层优化、预训练模型选择、模型蒸馏等6个方面进行模型调优，在中心采样评估方式下，Kinetics-400上精度较原论文实现提升3.95个点。更多细节请参考[PP-TSM模型解析](https://zhuanlan.zhihu.com/p/382134297)。

### PP-TSMv2

PP-TSMv2是轻量化的视频分类模型，基于CPU端模型[PP-LCNetV2](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-LCNetV2.md)进行优化，从骨干网络与预训练模型选择、数据增强、tsm模块调优、输入帧数优化、解码速度优化、DML蒸馏、LTA模块等7个方面进行模型调优，在中心采样评估方式下，精度达到75.16%，输入10s视频在CPU端的推理速度仅需456ms。更多细节参考[PP-TSMv2技术报告](./pp-tsm_v2.md)。


<a name="2"></a>
## 2. 性能benchmark

PP-TSMv2模型与主流模型之间CPU推理速度对比(按预测总时间排序)：

|模型名称 | 骨干网络 | 精度% | 预处理时间ms | 模型推理时间ms | 预测总时间ms |
| :---- | :---- | :----: |:----: |:----: |:----: |
| PP-TSM | MobileNetV2 |  68.09 | 52.62 | 137.03 | 189.65 |
| PP-TSM | MobileNetV3 |  69.84| 53.44 | 139.13 | 192.58 |
| **PP-TSMv2** | PP-LCNet_v2.8f | **72.45**| 53.37 | 189.62 | **242.99** |
| **PP-TSMv2** | PP-LCNet_v2.16f |	**75.16**|  68.07 | 388.64 | **456.71** |
| SlowFast | 4*16 |74.35 | 110.04 | 1201.36 | 1311.41 |
| TSM | R50 |  71.06 | 52.47 | 1302.49 | 1354.96 |
|PP-TSM	| R50 |	75.11 | 52.26  | 1354.21 | 1406.48 |
|*MoViNet | A0 | 66.62 | 148.30 |	1290.46 | 1438.76 |
|PP-TSM	| R101 |  76.35| 52.50 | 2236.94 | 2289.45 |
| TimeSformer |	base |	 77.29 | 297.33 |	14034.77 |	14332.11 |
| TSN | R50	| 69.81 | 860.41 | 18359.26 | 19219.68 |
| *VideoSwin | B | 82.4 | 76.21 | 32983.49 | 33059.70 |


* 注: 带`*`表示该模型未使用mkldnn进行预测加速。

更多细节请查看[benchmark](../../benchmark.md)文档。

<a name="3"></a>
## 3. 数据准备

Kinetics-400数据下载及准备请参考[Kinetics-400数据准备](../../dataset/k400.md)

UCF101数据下载及准备请参考[UCF-101数据准备](../../dataset/ucf101.md)

<a name="4"></a>
## 4. 模型训练

下面以Kinetics-400数据集为例，说明模型训练、测试、推理、压缩方法。

<a name="41"></a>
### 4.1 预训练模型下载

PP-TSM模型使用[PaddleClas ssld](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/advanced_tutorials/ssld.md)图像预训练模型作为Backbone初始化参数，各预训练模型下载链接如下：

|模型名称 | 骨干网络 | 预训练模型 |
| :---- | :---- | :----: |
| PP-TSMv2 | **LCNet_v2** |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNetV2_base_ssld_pretrained.pdparams) |
| PP-TSM | **ResNet50** | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams) |
| PP-TSM | MobileNetV2 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_ssld_pretrained.pdparams) |
| PP-TSM | MobileNetV3 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_0_ssld_pretrained.pdparams) |
| PP-TSM | ResNet101 | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ResNet101_vd_ssld_pretrained.pdparams) |

更多预训练模型下载链接可参考`paddlevideo/modeling/backbones/pptsm_xx.py`中各文件头部注释。

下载完成后，将文件路径添加到配置文件中的`MODEL.framework.backbone.pretrained`字段，如下：

```yaml
MODEL:
    framework: "Recognizer2D"
    backbone:
        name: "ResNetTweaksTSM"
        pretrained: 将路径填写到此处
```

<a name="42"></a>
### 4.2 多卡训练

PP-TSMv2在Kinetics400数据集使用8卡训练，多卡训练启动命令如下:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform.yaml
```

- 训练各参数含义参考[使用说明](../../usage.md)，若希望加速训练过程，可以按照使用说明第6章节开启混合精度训练。

- `batch_size`可以根据机器显存大小进行调整，请注意`batch_size`调整后学习率大小`learning rate`也需要按比例调整。


<a name="43"></a>
### 4.3 蒸馏训练

通过模型蒸馏将大模型的知识迁移到小模型中，可以进一步提升模型精度。PP-TSMv2基于DML蒸馏，teacher模型使用PP-TSM ResNet-50 backbone。蒸馏训练启动方式如下：

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform_dml_distillation.yaml
```

知识蒸馏更多细节参考[知识蒸馏](../../distillation.md)。


<a name="44"></a>
### 4.4 配置文件说明

PP-TSM模型提供的各配置文件均放置在[configs/recognition/pptsm](../../../../configs/recognition/pptsm)目录下，配置文件名按如下格式组织:

`模型名称_骨干网络名称_数据集名称_数据格式_测试方式_其它.yaml`。

- 数据格式包括`frame`和`video`，`video`表示使用在线解码的方式进行训练，`frame`表示先将视频解码成图像帧存储起来，训练时直接读取图片进行训练。使用不同数据格式，仅需修改配置文件中的`DATASET`和`PIPELINE`字段，参考[pptsm_k400_frames_uniform.yaml](../../../../configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml)和[pptsm_k400_videos_uniform.yaml](../../../../configs/recognition/pptsm/pptsm_k400_videos_uniform.yaml)。注意，由于编解码的细微差异，两种格式训练得到的模型在精度上可能会有些许差异。

- 测试方式包括`uniform`和`dense`，uniform表示中心采样，dense表示密集采样，更多细节参考第5章节模型测试部分。

- 您也可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的。

<a name="45"></a>
### 4.5 配置文件推荐使用

- 1. 数据格式：如硬盘存储空间足够，推荐使用`frame`格式，解码一次后，后续可以获得更快的训练速度。相较于使用视频格式训练，frame格式输入可以加快训练速度，加速比约4-5倍，但会占用更大的存储空间，如Kinetics-400数据集video格式135G，解码成图像后需要2T。

- 2. 测试方式：对于产业落地场景，推荐使用`uniform`方式，简洁高效，可以获得较好的精度与速度平衡。

- 3. 对于CPU或端侧需求，推荐使用`PP-TSMv2`，精度较高，速度快，具体性能和速度对比请查看[benchmark](../../benchmark.md)文档。PP-TSMv2提供8帧输入和16帧输入两套配置，8帧速度更快，精度稍低。16帧精度更高，速度稍慢。如果追求高精度，推荐使用16帧，配置文件为无蒸馏-[pptsm_lcnet_k400_16frames_uniform.yaml](../../../../configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform.yaml)，加蒸馏-[pptsm_lcnet_k400_16frames_uniform_dml_distillation.yaml](../../../../configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform_dml_distillation.yaml)。相对于无蒸馏，蒸馏后能获得更高的精度，但训练时需要更大的显存，以运行教师模型。如果对速度要求极高，推荐使用8帧，配置文件为无蒸馏-[pptsm_lcnet_k400_8frames_uniform.yaml](../../../../configs/recognition/pptsm/v2/pptsm_lcnet_k400_8frames_uniform.yaml)，加蒸馏-[pptsm_lcnet_k400_8frames_uniform_dml_distillation.yaml](../../../../configs/recognition/pptsm/v2/pptsm_lcnet_k400_8frames_uniform_dml_distillation.yaml)。

- 4. 对于GPU服务器端需求，推荐使用`PP-TSM`，对应配置文件为[pptsm_k400_frames_uniform.yaml](../../../../configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml)。GPU端推理，速度瓶颈更多在于数据预处理(视频编解码)部分，更优的解码器和更高的精度，会是侧重考虑的部分。

<a name="5"></a>
## 5. 模型测试

对于视频分类任务，模型测试时有两种不同的方式，`中心采样`(Uniform)和`密集采样`(Dense)。中心采样速度快，适合产业应用，但精度稍低。密集采样能进一步提升精度，但由于测试要对多个clip进行预测，比较耗时。轻量化模型PP-TSMv2统一使用中心采样方式进行评估。PP-TSM则提供两种不同的评估方式。

<a name="51"></a>
### 5.1 中心采样测试

中心采样测试，1个视频共采样1个clips。对输入视频，时序上，等分成`num_seg`段，每段中间位置采样1帧；空间上，中心位置采样。对Uniform采样方式，PP-TSM模型在训练时同步进行测试，您可以通过在训练日志中查找关键字`best`获取模型测试精度，日志示例如下:

```txt
Already save the best model (top1 acc)0.7467
```

也可以使用如下命令对训练好的模型进行测试：
```bash
python3 main.py --test -c configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform_dml_distillation.yaml -w output/PPTSMv2/PPTSMv2_best.pdparams
```

<a name="52"></a>
### 5.2 密集采样测试

密集采样测试，1个视频共采样`10*3=30`个clips。时序上，先等分成10个片段，每段从起始位置开始，以`64//num_seg`为间隔连续采样`num_seg`帧；空间上，左中，中心，右中3个位置采样。对Dense采样方式，需要在训练完成后单独运行测试代码，其启动命令如下：

```bash
python3 main.py --test -c configs/recognition/pptsm/pptsm_k400_frames_dense.yaml -w output/ppTSM/ppTSM_best.pdparams
```

- 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。


<a name="6"></a>
## 6. 模型推理

<a name="61"></a>
### 导出推理模型

```bash
python3.7 tools/export_model.py -c configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform_dml_distillation.yaml \
                                -p output/PPTSMv2/PPTSMv2_best.pdparams \
                                -o inference/PPTSMv2
```

上述命令会在`inference/PPTSMv2`下生成预测所需的文件，结构如下:
```
├── inference/PPTSMv2
│   ├── PPTSMv2.pdiparams       # 模型权重文件
│   ├── PPTSMv2.pdiparams.info  # 模型信息文件
│   └── PPTSMv2.pdmodel           # 模型结构文件
```

<a name="62"></a>
### 基于python预测引擎推理

运行下面命令，对示例视频文件`data/example.avi`进行分类:
```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/pptsm/v2/pptsm_lcnet_k400_16frames_uniform_dml_distillation.yaml \
                           --model_file inference/PPTSMv2/PPTSMv2.pdmodel \
                           --params_file inference/PPTSMv2/PPTSMv2.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```


输出示例如下:

```
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 1.0
```


可以看到，使用在Kinetics-400上训练好的PP-TSMv2模型对`data/example.avi`进行预测，输出的top1类别id为`5`，置信度为1.0。通过查阅类别id与名称对应表`data/k400/Kinetics-400_label_list.txt`，可知预测类别名称为`archery`。

<a name="63"></a>
### 基于c++预测引擎推理

PaddleVideo 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端C++预测](../../../../deploy/cpp_infer/)来完成相应的推理部署。


<a name="64"></a>
### 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考[Paddle Serving](https://github.com/PaddlePaddle/Serving) 代码仓库。

PaddleVideo 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[基于python的模型服务化部署](../../../../deploy/python_serving/)或[基于c++的模型服务化部署](../../../../deploy/cpp_serving/)来完成相应的部署工作。


<a name="65"></a>
### Paddle2ONNX 模型预测与转换

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) 代码仓库。

PaddleVideo 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](../../../../deploy/paddle2onnx/)来完成相应的部署工作。


<a name="7"></a>
## 7. 模型库下载

在Kinetics-400数据集上模型效果:

| 模型名称 | 骨干网络 | 测试方式 | 采样帧数 | Top-1% | 训练模型 |
| :------: | :----------: | :----: | :----: | :----: | :---- |
| PP-TSMv2 | LCNet_v2 |  Uniform | 8 | 71.81 | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/PPTSMv2_k400_8f.pdparams) |
| PP-TSMv2 | LCNet_v2 |  Uniform | 16 | 73.1 | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/PPTSMv2_k400_16f.pdparams) |
| PP-TSM | MobileNetV2 |  Uniform | 8 | 68.09 | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_mv2_k400.pdparams) |
| PP-TSM | MobileNetV3 |  Uniform | 8 | 69.84 | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_mv3_k400.pdparams) |
| PP-TSM | ResNet50 |  Uniform | 8 | 74.54 | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_uniform.pdparams) |
| PP-TSM | ResNet50 |  Dense | 8 | 75.69 | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense.pdparams) |
| PP-TSM | ResNet101 | Dense | 8 | 77.15 | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSM_k400_dense_r101.pdparams) |


蒸馏模型:

| 模型名称 | 骨干网络 | 蒸馏方式 | 测试方式 | 采样帧数 | Top-1% | 训练模型 |
| :------: | :----------: | :----: | :----: | :----: | :---- | :---- |
| PP-TSMv2 | LCNet_v2 | DML | Uniform | 8 | 72.45 | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/PPTSMv2_k400_8f_dml.pdparams) \| [Student模型](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/PPTSMv2_k400_8f_dml_student.pdparams) |
| PP-TSMv2 | LCNet_v2 | DML | Uniform | 16 | 75.16 | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/PPTSMv2_k400_16f_dml.pdparams) \| [Student模型](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/PPTSMv2_k400_16f_dml_student.pdparams) |
| PP-TSM | ResNet50 | KD | Uniform | 8 | 75.11 | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_uniform_distill.pdparams) |
| PP-TSM | ResNet50 | KD | Dense | 8 | 76.16 | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense_distill.pdparams) |
| PP-TSM | ResNet101 | KD | Uniform | 8 | 76.35 | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSM_k400_uniform_distill_r101.pdparams) |


## 参考论文

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), Geoffrey Hinton, Oriol Vinyals, Jeff Dean
