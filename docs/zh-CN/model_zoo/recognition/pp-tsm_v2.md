# PP-TSMv2
---
## 目录

- [1. 模型简介](#1)
- [2. 模型细节](#2)
    - [2.1 骨干网络与预训练模型选择](#21)
    - [2.2 数据增强](#22)
    - [2.3 tsm模块调优](#23)
    - [2.4 输入帧数优化](#24)
    - [2.5 解码速度优化](#25)
    - [2.6 DML蒸馏](#26)
    - [2.7 LTA模块](#27)  
- [3. 快速体验](#3)
- [4. 模型训练、压缩、推理部署](#4)


<a name="1"></a>
## 1. 模型简介

视频分类任务是指输入视频，输出标签类别。如果标签都是行为类别，则该任务也称为行为识别。随着AI在各个行业的应用普及，工业及体育场景下对轻量化行为识别模型的需求日益增多，为此我们提出了高效的轻量化行为识别模型PP-TSMv2。

PP-TSMv2沿用了部分PP-TSM的优化策略，从骨干网络与预训练模型选择、数据增强、tsm模块调优、输入帧数优化、解码速度优化、DML蒸馏、LTA模块等7个方面进行模型调优，在中心采样评估方式下，精度达到75.16%，输入10s视频在CPU端的推理速度仅需456ms。


<a name="2"></a>
## 2. 模型细节

<a name="21"></a>
### 2.1 骨干网络与预训练模型选择

在骨干网络的选择上，PP-TSMv2选用了针对基于CPU端设计的轻量化骨干网络[PP-LCNetV2](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/PP-LCNetV2.md)。PP-LCNetV2通过组合PW卷积、SE模块、Shortcut和Reparameterizatio等策略，在不使用额外数据的前提下，在图像分类 ImageNet 数据集上的性能如下表所示。

| Model | Top-1 Acc(\%) | Latency(ms) |
|:--:|:--:|:--:|
| MobileNetV3_Large_x1_25 | 76.4 | 5.19 |
| PPLCNetV2_base | 77.04 | 4.32 |
| PPLCNetV2_base_ssld | 80.07| 4.32 |

在预训练模型选择上，我们以使用[SSLD](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/advanced_tutorials/knowledge_distillation.md)在 ImageNet上蒸馏得到的模型作为预训练模型，相较于未使用ssld的预训练模型，其提升效果如下表所示。

| 策略 | Top-1 Acc(\%) |
|:--:|:--:|
| baseline | 69.06 |
| baseline + SSLD Backbone | 69.95(+**0.89**) |

<a name="22"></a>
### 2.2 数据增强

我们沿用了PP-TSM使用数据增强策略VideoMix，将两个视频以一定的权值叠加构成新的输入样本。具体的，对每个视频，首先抽取固定数量的帧，并给每一帧赋予相同的权重，然后与另一个视频叠加作为新的输入视频。这种方式对精度的提升效果如下表所示。

| 策略 | Top-1 Acc(\%) |
|:--:|:--:|
| baseline | 69.06 |
| baseline + VideoMix | 69.36(+**0.3**) |

<a name="23"></a>
### 2.3 tsm模块调优

在骨干网络的基础上，我们添加了时序位移模块提取时序信息。对于插入位置，TSM原论文中将temporal_shift模块插入残差结构之中，但PP-LCNetV2为了加快模型速度，去除了部分残差连接。PP-LCNetV2整体结构分为4个stage，我们实验探索了时序位移模块最佳插入位置。对于插入数量，temporal_shift模块会加大模型的运行时间，我们探索了其最优插入数量，实验结果如下表所示。

| 策略 | Top-1 Acc(\%) |
|:--:|:--:|
| baseline | 69.06 |
| baseline + tsm in stage1 | 69.84 |
| baseline + tsm in stage2 | 69.84 |
| **baseline + tsm in stage3** | **70.02(+0.96)** |
| baseline + tsm in stage4 | 69.98 |
| baseline + tsm in stage1,2 | 69.77 |
| baseline + tsm in stage3,4 | 70.05 |
| baseline + tsm in stage1,2,3,4 | 70.06 |

可以看到，在高层插入时序位移模块的效果优于低层。在Stage3中插入1个temporal_shift模块，能达到精度和速度上的最优。

<a name="24"></a>
### 2.4 输入帧数优化

对于10s的视频，我们会抽取一定数量的帧输入到网络中。PP-TSMv2采样分段采样策略，即先将视频按时间长度等分成N段，然后在每段中随机选取一帧，组合得到N帧作为模型输入。

输入帧数的增加一定程度上能提升模型精度，但同时会带来数据预处理及模型推理时间的显著增加。综合考虑性能和速度，我们采用16帧作为输入，相较于8帧输入，精度提升效果如下表所示。

| 策略 | Top-1 Acc(\%) |
|:--:|:--:|
| baseline | 69.06 |
| baseline + 16f | 69.78(+**0.72**) |

<a name="25"></a>
### 2.5 解码速度优化

在解码速度上，我们对比了常见的视频解码库在视频分段采样策略中的速度，[测试数据](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/time-test.tar)，不同解码库速度对比如下表所示。PP-TSMv2最终选用[decord](https://github.com/dmlc/decord)作为解码器。

| lib | Time/s |
|:--:|:--:|
| opencv | 0.056 |
| **decord** | **0.043** |
| PyAV| 0.045 |

- 实用tips，若使用opencv进行解码，代码作如下优化能极大提升解码速度:

```python
    cap = cv2.VideoCapture(file_path)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 1. decode all frames, time cost!
    sampledFrames = []
    for i in range(videolen):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        sampledFrames.append(img)

    cap.release()

    # 2. get frame index
    frames_idx = [xxx]

    # 3. sample
    frames = np.array(sampledFrames)
    imgs = []
    for idx in frames_idx:
        imgbuf = frames[idx]
        img = Image.fromarray(imgbuf, mode='RGB')
        imgs.append(img)
```

优化后:
```python
    cap = cv2.VideoCapture(file_path)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 1. get frame index
    frames_idx = [xxx]

    # 2. decode target frame
    imgs = []
    for i in range(videolen):
        ret = cap.grab()
        # maybe first frame is empty
        if ret == False:
            continue  
        if frames_idx and i == frames_idx[0]:
            frames_idx.pop(0)
            ret, frame = cap.retrieve()
            if frame is None:
                break
            imgbuf = frame[:, :, ::-1]
            img = Image.fromarray(imgbuf, mode='RGB')
            imgs.append(img)
        if frames_idx == None:
            break
    cap.release()
```

<a name="26"></a>
### 2.6 DML蒸馏

通过[模型蒸馏](../../distillation.md)将大模型的知识迁移到小模型中，可以进一步提升模型精度。PP-TSMv2使用[DML蒸馏](https://arxiv.org/pdf/1706.00384.pdf)方法，在蒸馏的过程中，不依赖于教师模型，两个结构相同的模型互相学习，计算彼此输出（logits）的KL散度，最终完成训练过程。PP-TSMv2优化过程中，分别尝试了以自身或者以PP-TSM ResNet-50 backbone作为教师模型进行蒸馏，性能提升效果如下表。

| 策略 | 教师模型 | Top-1 acc |
| --- | --- | --- |
| baseline | - | 69.06% |
| DML | PP-TSMv2 | 70.34%(**+1.28%**) |
| DML | PP-TSM_ResNet50 | 71.27%(**+2.20%**) |


<a name="27"></a>
### 2.7 LTA模块

temporal shift模块通过把特征在时间通道上位移，获取时序信息。但这种位移方式仅让局部的特征进行交互，缺少对全局时序信息的建模能力。为此我们提出了轻量化的时序attention模块(Lightweight Temporal Attention, LTA)，如图所示，通过全局池化组合可学习的fc层，得到全局尺度上的时序attention。在tsm模块之前，添加时序attention模块，使得网络在全局信息的指导下进行时序位移。LTA模块能够在基本不增加推理时间的前提下，进一步提升模型精度。

<div align="left">
  <img src="https://user-images.githubusercontent.com/22365664/209295833-3f68ab67-c7e4-460f-ad12-68d4b9115460.png" width="450px"/><br>
</div>

| 策略 | Top-1 Acc(\%) |
|:--:|:--:|
| pptsmv2 w/o temporal_attention | 74.38 |
| pptsmv2 w/ temporal_attention | 75.16(+**0.78**) |

<a name="3"></a>
## 3. 快速体验

参考[快速开始文档](../../quick_start.md)，安装`ppvideo` 2.3.0版本，即可快速体验使用PP-TSMv2模型进行预测。

<a name="4"></a>
## 4. 模型训练、压缩、推理部署


更多教程，包括模型训练、模型压缩、推理部署等，请参考[使用文档](./pp-tsm.md)。
