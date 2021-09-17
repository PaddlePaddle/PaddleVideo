[English](README.md) | 中文

# PaddleVideo

## 简介

![python version](https://img.shields.io/badge/python-3.7+-orange.svg) ![paddle version](https://img.shields.io/badge/PaddlePaddle-2.0-blue)


PaddleVideo是[飞桨官方](https://www.paddlepaddle.org.cn/?fr=paddleEdu_github)出品的视频模型开发套件，旨在帮助开发者更好的进行视频领域的学术研究和产业实践。

<div align="center">
  <img src="docs/images/home.gif" width="450px"/><br>
</div>

### **如果本项目对您有帮助，欢迎点击页面右上方star⭐，方便访问**


## 特性

- **更多的数据集和模型结构**
    PaddleVideo 支持更多的数据集和模型结构，包括[Kinetics400](docs/zh-CN/dataset/k400.md)，UCF101，YoutTube8M，NTU-RGB+D等数据集，模型结构涵盖了视频分类模型TSN，TSM，SlowFast，TimeSformer，AttentionLSTM，ST-GCN和视频定位模型[BMN](./docs/zh-CN/model_zoo/localization/bmn.md)等。

- **更高指标的模型算法**
    PaddleVideo 提供更高精度的模型结构解决方案，在基于TSM标准版改进的[PP-TSM](docs/zh-CN/model_zoo/recognition/pp-tsm.md)上，在Kinectics400数据集上达到2D网络SOTA效果，Top1 Acc 76.16% 相较标准版TSM模型参数量持平，且取得更快的模型速度。

- **更快的训练速度**
    PaddleVideo 提供更快速度的训练阶段解决方案，包括[混合精度训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)，分布式训练，针对Slowfast模型的Multigrid训练策略，OP融合策略以及更快的数据预处理模块等。

- **全流程可部署**
    PaddleVideo 提供全流程的预测部署方案，支持PaddlePaddle2.0动转静功能，方便产出可快速部署的模型，完成部署阶段最后一公里。

- **丰富的应用案例**
    PaddleVideo 提供了基于行为识别和动作检测技术的多个实用案例，包括[FootballAction](https://github.com/PaddlePaddle/PaddleVideo/tree/application/FootballAction)和VideoTag。


### 模型性能概览


| 领域               |                             模型                             |                       数据集                       | 精度指标 |   精度%   |
| :----------------- | :----------------------------------------------------------: | :------------------------------------------------: | :------: | :-------: |
| 行为识别|   [**PP-TSM**](./docs/zh-CN/model_zoo/recognition/pp-tsm.md)  |    [Kinetics-400](./docs/zh-CN/dataset/k400.md)    |  Top-1   | **76.16** |
| 行为识别 |  [**PP-TSN**](./docs/zh-CN/model_zoo/recognition/pp-tsn.md)  |    [Kinetics-400](./docs/zh-CN/dataset/k400.md)    |  Top-1   | **75.06** |
| 行为识别 | [AGCN](./docs/zh-CN/model_zoo/recognition/pp-agcn.md) | [FSD-10](./docs/zh-CN/dataset/fsd10.md) | Top-1 | 90.66 |
| 行为识别 | [ST-GCN](./docs/zh-CN/model_zoo/recognition/stgcn.md) | [FSD-10](./docs/zh-CN/dataset/fsd10.md) | Top-1 | 86.66 |
| 行为识别 | [TimeSformer](./docs/zh-CN/model_zoo/recognition/timesformer.md) |    [Kinetics-400](./docs/zh-CN/dataset/k400.md)    |  Top-1   |   77.29   |
| 行为识别 |  [SlowFast](./docs/zh-CN/model_zoo/recognition/slowfast.md)  |    [Kinetics-400](./docs/zh-CN/dataset/k400.md)    |  Top-1   |   75.84   |
| 行为识别 |       [TSM](./docs/zh-CN/model_zoo/recognition/tsm.md)       |    [Kinetics-400](./docs/zh-CN/dataset/k400.md)    |  Top-1   |   71.06   |
| 行为识别 |       [TSN](./docs/zh-CN/model_zoo/recognition/tsn.md)       |    [Kinetics-400](./docs/zh-CN/dataset/k400.md)    |  Top-1   |   69.81   |
| 行为识别 | [AttentionLSTM](./docs/zh-CN/model_zoo/recognition/attention_lstm.md) |  [Youtube-8M](./docs/zh-CN/dataset/youtube8m.md)   |  Hit@1   |   89.0    |
| 视频动作定位   |      [BMN](./docs/zh-CN/model_zoo/localization/bmn.md)       | [ActivityNet](./docs/zh-CN/dataset/ActivityNet.md) |   AUC    |   67.23   |

<a name="欢迎加入PaddleVideo技术交流群"></a>
## 欢迎加入PaddleVideo技术交流群
- 微信扫描二维码添加运营同学，回复 **“视频”**，即可邀请您加入官方交流群，获得更高效的问题答疑，与各行各业开发者充分交流，期待您的加入。

<div align="center">
<img src="./docs/images/joinus.PNG"  width = "200" height = "200" />
</div>

## 特色应用方案效果
- [特色应用01: 大规模视频3k类标签方案VideoTag](https://github.com/PaddlePaddle/PaddleVideo/tree/application/VideoTag)

<div align="center">
  <img src="docs/images/VideoTag.gif" width="450px"/><br>
</div>

- [特色应用02: 足球动作定位方案FootballAction](https://github.com/PaddlePaddle/PaddleVideo/tree/application/FootballAction)

<div align="center">
  <img src="docs/images/FootballAction.gif" width="450px"/><br>
</div>


## 文档教程
- 免费视频课程、PPT、AIStudio教程
    - [飞桨视频库全面解析](https://aistudio.baidu.com/aistudio/course/introduce/6742)
    - [视频分类及动作识别介绍](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/tutorials/summarize.md)
    - [【官方】Paddle 2.1实现视频理解经典模型 - TSN](https://aistudio.baidu.com/aistudio/projectdetail/2250682)
    - [【官方】Paddle 2.1实现视频理解经典模型 - TSM](https://aistudio.baidu.com/aistudio/projectdetail/2310889)
    - [BMN视频动作定位](https://aistudio.baidu.com/aistudio/projectdetail/2250674)
- 快速入门
    - [安装说明](docs/zh-CN/install.md)
    - [快速开始](docs/zh-CN/start.md)
- 代码组织
    - [模型库设计思路详解](docs/zh-CN/tutorials/modular_design.md)
    - [配置模块参数详解](docs/zh-CN/tutorials/config.md)
- 丰富的模型库
    - [视频分类](docs/zh-CN/model_zoo/README.md)
       - [TSN](docs/zh-CN/model_zoo/recognition/tsn.md)
       - [TSM](docs/zh-CN/model_zoo/recognition/tsm.md)
       - [PP-TSM](docs/zh-CN/model_zoo/recognition/pp-tsm.md)
       - [PP-TSN](docs/zh-CN/model_zoo/recognition/pp-tsn.md)
       - [SlowFast](docs/zh-CN/model_zoo/recognition/slowfast.md)
       - [TimeSformer](docs/zh-CN/model_zoo/recognition/timesformer.md)
       - [Attention-LSTM](docs/zh-CN/model_zoo/recognition/attention_lstm.md)
    - [动作定位](docs/zh-CN/model_zoo/README.md)
       - [BMN](docs/zh-CN/model_zoo/localization/bmn.md)
    - [基于骨骼的行为识别](docs/zh-CN/model_zoo/README.md)
       - [ST-GCN](docs/zh-CN/model_zoo/recognition/stgcn.md)
       - [AGCN](docs/zh-CN/model_zoo/recognition/agcn.md)
    - 时空动作检测 <sup>coming soon</sup>
    - ActBERT: 自监督多模态视频文字学习<sup>coming soon</sup>
- 项目实战
    - [PP-TSM实践](docs/zh-CN/tutorials/pp-tsm.md)
    - [训练加速](docs/zh-CN/tutorials/accelerate.md)
    - [预测部署](docs/zh-CN/tutorials/deployment.md)
- 辅助工具
    - [benchmark](docs/zh-CN/benchmark.md)
    - [工具](docs/zh-CN/tools.md)
- [技术交流群](#欢迎加入PaddleVideo技术交流群)
- [特色详解](#特色详解)
- [许可证书](#许可证书)
- [贡献代码](#贡献代码)

## 特色详解

- **丰富的模型种类**  
    PaddleVideo包含视频分类和动作定位方向的多个主流领先模型，其中TSN, TSM和SlowFast是End-to-End的视频分类模型，Attention LSTM是比较流行的视频特征序列模型，BMN是视频动作定位模型。TSN是基于2D-CNN的经典解决方案，TSM是基于时序移位的简单高效视频时空建模方法，SlowFast是FAIR在ICCV2019提出的3D视频分类模型，特征序列模型Attention LSTM速度快精度高。BMN模型是百度自研模型，为2019年ActivityNet夺冠方案。基于百度飞桨产业实践，我们自研并开源了PP-TSM，该模型基于TSM进行优化，在保持模型参数量和计算量不增加的前提下，精度得到大幅提升。同时，我们的通用优化策略可以广泛适用于各种视频模型，未来我们将进行更多的模型优化工作，比如TSN、SlowFast、X3D等，敬请期待。  

- **3000分类预训练模型**  
    飞桨大规模视频分类模型VideoTag基于百度短视频业务千万级数据，支持3000个源于产业实践的实用标签，具有良好的泛化能力，非常适用于国内大规模（千万/亿/十亿级别）短视频分类场景的应用。VideoTag采用两阶段建模方式，即图像建模和序列学习。第一阶段，使用少量视频样本（十万级别）训练大规模视频特征提取模型(Extractor)；第二阶段，使用千万级数据训练预测器(Predictor)，最终实现在超大规模（千万/亿/十亿级别）短视频上产业应用。  

- **SOTA算法PP-TSM**  
    与图像任务相比，视频任务的难点在于时序信息的提取。传统的2D网络难以捕获时序信息，通过增加时序通道，3D网络能更好的联合时序特征建模。但3D网络的计算量较大，部署成本较高。TSM模型通过时序位移模块，有效平衡了计算效率和模型的性能，是一种高效实用视频理解模型，在工业界广泛应用。PaddleVideo基于飞桨框架2.0对TSM进行改进，在不增加参数量和计算量的情况下，在多个数据集上精度显著超过TSM论文精度，在仅用ImageNet pretrain情况下，PP-TSM在Kinetics400数据集top1分别达到76.16%，是至今为止开源的2D视频模型中在相同条件下的最高性能。

- **更快的训练速度**  
    视频任务相比于图像任务的训练往往更加耗时，其原因主要有两点: 一是模型上，视频任务使用的模型通常有更大的参数量与计算量；一是数据上，视频文件解码通常极为耗时。为优化视频模型训练速度，项目中分别从模型角度和数据预处理角度，实现了多种视频训练加速方案。针对TSM模型，通过op融合的方式实现了temporal shift op，在节省显存的同时加速训练过程。针对TSN模型，实现了基于DALI的纯GPU解码方案，训练速度较标准实现加速3.6倍。针对SlowFast模型，结合Decode解码库和DataLoader多子进程异步加速，训练速度较原始实现提升100%，使用Multigrid策略训练总耗时可以进一步减少。预先解码存成图像的方案也能显著加速训练过程，TSM/PP-TSM在训练全量Kinetics-400数据集80个epoch只需要2天。  

## 赛事支持
[CCKS 2021：知识增强的视频语义理解](https://www.biendata.xyz/competition/ccks_2021_videounderstanding/)

## 许可证书
本项目的发布受[Apache 2.0 license](LICENSE)许可认证。


## 欢迎贡献
我们欢迎您的任何贡献并感谢您的支持，更多信息请参考 [contribution guidelines](docs/CONTRIBUTING.md).

- 非常感谢 [mohui37](https://github.com/mohui37) 贡献预测相关代码

## [需求征集](https://github.com/PaddlePaddle/PaddleVideo/issues/68)
