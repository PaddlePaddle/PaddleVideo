[English](README.md) | 中文

# PaddleVideo
## 最新动态
- 🔥🔥🔥 2021年CCF大数据与计算智能大赛火热进行中，欢迎参加CCF和百度飞桨联合推出奖金10万元的赛题[基于飞桨实现花样滑冰选手骨骼点动作识别](https://www.datafountain.cn/competitions/519)！:tada:
赛题baseline由PaddleVideo提供[ST-GCN](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/stgcn.md)，
[教程链接](https://aistudio.baidu.com/aistudio/projectdetail/2417717)，[视频链接](https://www.bilibili.com/video/BV1w3411172G)

## 简介

![python version](https://img.shields.io/badge/python-3.7+-orange.svg) ![paddle version](https://img.shields.io/badge/PaddlePaddle-2.0-blue)


PaddleVideo是[飞桨官方](https://www.paddlepaddle.org.cn/?fr=paddleEdu_github)出品的视频模型开发套件，旨在帮助开发者更好的进行视频领域的学术研究和产业实践。

<div align="center">
  <img src="docs/images/home.gif" width="450px"/><br>
</div>

### **⭐如果本项目对您有帮助，欢迎点击页面右上方star~ ⭐**


## 特性

- **更多的数据集和模型结构**
    PaddleVideo 支持更多的数据集和模型结构，包括[Kinetics400](docs/zh-CN/dataset/k400.md)，UCF101，YoutTube8M，NTU-RGB+D等数据集，模型结构涵盖了视频分类模型TSN，TSM，SlowFast，TimeSformer，AttentionLSTM，ST-GCN和视频定位模型BMN等。

- **更高指标的模型算法**
    PaddleVideo 提供更高精度的模型结构解决方案，在基于TSM标准版改进的[PP-TSM](docs/zh-CN/model_zoo/recognition/pp-tsm.md)上，在Kinectics400数据集上达到2D网络SOTA效果，Top1 Acc 76.16% 相较标准版TSM模型精度提升5%以上，参数量持平，且取得更快的模型速度。

- **更快的训练速度**
    PaddleVideo 提供更快速度的训练阶段解决方案，包括混合精度训练，分布式训练，针对Slowfast模型的Multigrid训练策略，OP融合策略以及更快的数据预处理模块等。

- **全流程可部署**
    PaddleVideo 提供全流程的预测部署方案，支持PaddlePaddle2.0动转静功能，方便产出可快速部署的模型，完成部署阶段最后一公里。

- **丰富的应用案例**
    PaddleVideo 提供了基于行为识别和动作检测技术的多个实用案例，包括FootballAction和VideoTag。


### 模型性能概览


| 领域               |                             模型                             |                       数据集                       | 精度指标 |   精度%   |
| :----------------- | :----------------------------------------------------------: | :------------------------------------------------: | :------: | :-------: |
| 行为识别|   [**PP-TSM**](./docs/zh-CN/model_zoo/recognition/pp-tsm.md)  |    [Kinetics-400](./docs/zh-CN/dataset/k400.md)    |  Top-1   | **76.16** |
| 行为识别 |  [**PP-TSN**](./docs/zh-CN/model_zoo/recognition/pp-tsn.md)  |    [Kinetics-400](./docs/zh-CN/dataset/k400.md)    |  Top-1   | **75.06** |
| 行为识别 | [**PP-TimeSformer**](./docs/zh-CN/model_zoo/recognition/pptimesformer.md) |    [Kinetics-400](./docs/zh-CN/dataset/k400.md)    |  Top-1   |   79.49   |
| 行为识别 | [AGCN](./docs/zh-CN/model_zoo/recognition/agcn.md) | [FSD](./docs/zh-CN/dataset/fsd.md) | Top-1 | 62.29 |
| 行为识别 | [ST-GCN](./docs/zh-CN/model_zoo/recognition/stgcn.md) | [FSD](./docs/zh-CN/dataset/fsd.md) | Top-1 | 59.07 |
| 行为识别 | [TimeSformer](./docs/zh-CN/model_zoo/recognition/timesformer.md) |    [Kinetics-400](./docs/zh-CN/dataset/k400.md)    |  Top-1   |   77.29   |
| 行为识别 |  [SlowFast](./docs/zh-CN/model_zoo/recognition/slowfast.md)  |    [Kinetics-400](./docs/zh-CN/dataset/k400.md)    |  Top-1   |   75.84   |
| 行为识别 |       [TSM](./docs/zh-CN/model_zoo/recognition/tsm.md)       |    [Kinetics-400](./docs/zh-CN/dataset/k400.md)    |  Top-1   |   71.06   |
| 行为识别 |       [TSN](./docs/zh-CN/model_zoo/recognition/tsn.md)       |    [Kinetics-400](./docs/zh-CN/dataset/k400.md)    |  Top-1   |   69.81   |
| 行为识别 | [AttentionLSTM](./docs/zh-CN/model_zoo/recognition/attention_lstm.md) |  [Youtube-8M](./docs/zh-CN/dataset/youtube8m.md)   |  Hit@1   |   89.05    |
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
    - [花样滑冰选手骨骼点动作识别ST-GCN教程](https://aistudio.baidu.com/aistudio/projectdetail/2417717)
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
       - [PP-TimeSformer](docs/zh-CN/model_zoo/recognition/pptimesformer.md)
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
- [赛事支持](#赛事支持)
- [许可证书](#许可证书)
- [贡献代码](#贡献代码)


## 赛事支持
- [CCKS 2021：知识增强的视频语义理解](https://www.biendata.xyz/competition/ccks_2021_videounderstanding/)
- [基于飞桨实现花样滑冰选手骨骼点动作识别大赛](https://aistudio.baidu.com/aistudio/competition/detail/115/0/introduction)

## 许可证书
本项目的发布受[Apache 2.0 license](LICENSE)许可认证。


## 欢迎贡献
我们欢迎您的任何贡献并感谢您的支持，更多信息请参考 [contribution guidelines](docs/CONTRIBUTING.md).

- 非常感谢 [mohui37](https://github.com/mohui37)、[zephyr-fun](https://github.com/zephyr-fun)、[voipchina](https://github.com/voipchina) 贡献相关代码
