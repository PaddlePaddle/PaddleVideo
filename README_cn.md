[English](README.md) | 中文

# PaddleVideo

## 简介

![python version](https://img.shields.io/badge/python-3.7+-orange.svg) ![paddle version](https://img.shields.io/badge/PaddlePaddle-2.0-blue)


PaddleVideo飞桨视频模型开发套件，旨在帮助开发者更好的进行视频领域的学术研究和产业实践。

## 特性

- **模块化设计**
    PaddleVideo 基于统一的视频理解框架，使用模块化设计，将各部分功能拆分到不同组件中进行解耦。可以轻松的组合、配置和自定义组件来快速实现视频算法模型。

- **更多的数据集和模型结构**
    PaddleVideo 支持更多的数据集和模型结构，包括Kinectics400，ucf101，YoutTube8M等数据集，模型结构涵盖了视频分类模型TSN，TSM，SlowFast，AttentionLSTM和视频定位模型BMN等。

- **更高指标的模型算法**
    PaddleVideo 提供更高精度的模型结构解决方案，在基于TSM标准版改进的PPTSM上，达到2D网络SOTA效果，Top1 Acc 73.5% 相较标准版TSM提升3%且模型参数量持平。

- **更快的训练速度**
    PaddleVideo 提供更快速度的训练阶段解决方案，在SlowFast标准版上，训练速度相较于pytorch提速100%。

- **全流程可部署**
    PaddleVideo 提供全流程的预测部署方案，支持PaddlePaddle2.0动转静功能，方便产出可快速部署的模型，完成部署阶段最后一公里。

### 套件结构概览

<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Architectures</b>
      </td>
      <td>
        <b>Frameworks</b>
      </td>
      <td>
        <b>Components</b>
      </td>
      <td>
        <b>Data Augmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul><li><b>Recognition</b></li>
          <ul>
            <li>TSN</li>
            <li>TSM</li>
            <li>SlowFast</li>
            <li>PPTSM</li>
            <li>VideoTag</li>
            <li>AttentionLSTM</li>
          </ul>
        </ul>
        <ul><li><b>Localization</b></li>
          <ul>
            <li>BMN</li>
          </ul>
        </ul>
      </td>
      <td>
          <li>Recognizer1D</li>
          <li>Recognizer2D</li>
          <li>Recognizer3D</li>
          <li>Localizer</li>
        <HR></HR>
        <ul>Backbone
            <li>resnet</li>
            <li>resnet_tsm</li>
            <li>resnet_tweaks_tsm</li>
            <li>bmn</li>
        </ul>
        <ul>Head
            <li>tsm_head</li>
            <li>tsn_head</li>
            <li>bmn_head</li>
            <slowfast_head></li>
            <bmn_head></li>
        </ul>
      </td>
      <td>
        <ul><li><b>Solver</b></li>
          <ul><li><b>Optimizer</b></li>
              <ul>
                <li>Momentum</li>
                <li>RMSProp</li>
              </ul>
          </ul>
          <ul><li><b>LearningRate</b></li>
              <ul>
                <li>PiecewiseDecay</li>
              </ul>
          </ul>
        </ul>
        <ul><li><b>Loss</b></li>
          <ul>
            <li>CrossEntropy</li>
            <li>BMNLoss</li>  
          </ul>  
        </ul>  
        <ul><li><b>Metrics</b></li>
          <ul>
            <li>CenterCrop</li>
            <li>MultiCrop</li>  
          </ul>  
        </ul>
      </td>
      <td>
        <ul><li><b>Batch</b></li>
          <ul>
            <li>Mixup</li>
            <li>Cutmix</li>  
          </ul>  
        </ul>
        <ul><li><b>Image</b></li>
            <ul>
                <li>Resize</li>  
                <li>Flipping</li>  
                <li>MultiScaleCrop</li>
                <li>Crop</li>
                <li>Color Distort</li>  
                <li>Random Crop</li>
            </ul>
         </ul>
         <ul><li><b>Image</b></li>
            <ul>
                <li>Mixup </li>
                <li>Cutmix </li>
            </ul>
        </ul>  
      </td>  
    </tr>


</td>
    </tr>
  </tbody>
</table>

### 模型性能概览

视频分类模型在Kinectics-400数据集上Acc Top1精度和单卡Tesla V100上预测速度(VPS)对比图。

<div align="center">
  <img src="docs/images/acc_vps.jpeg" />
</div>

**说明：**
- 红色文字描述为PaddleVideo提供的模型，黑色文字描述为Pytorch实现
- PPTSM在TSM标准版上精度提升3.5%，预测速度也略有增加。
- 图中红色描述的模型均可在[模型库](#模型库)中获取

<a name="欢迎加入PaddleVideo技术交流群"></a>
## 欢迎加入PaddleVideo技术交流群
- 微信扫描二维码加入官方交流群，获得更高效的问题答疑，与各行各业开发者充分交流，期待您的加入。

<div align="center">
<img src="./docs/images/joinus.PNG"  width = "200" height = "200" />
</div>


## 文档教程

### 入门教程

- [安装说明](docs/zh-CN/install.md)
- [快速开始](docs/zh-CN/start.md)
- [benchmark](docs/zh-CN/benchmark.md)
- [工具](docs/zh-CN/tools.md)

### 进阶教程
- [模型库整体设计](docs/zh-CN/tutorials/modular_design.md)
- [配置模块设计](docs/zh-CN/tutorials/config.md)
- [PPTSM实践](docs/zh-CN/tutorials/pp-tsm.md)
- [训练加速方案](docs/zh-CN/tutorials/accelerate.md)
- [预测部署](docs/zh-CN/tutorials/deployment.md)
- [自定义开发]() <sup>coming soon</sup>

### 模型库

- 视频分类 [介绍](docs/zh-CN/model_zoo/README.md)
    - [Attention-LSTM](docs/zh-CN/model_zoo/recognition/attention_lstm.md)
    - [TSN](docs/zh-CN/model_zoo/recognition/tsn.md)
    - [TSM](docs/zh-CN/model_zoo/recognition/tsm.md)
    - [PPTSM](docs/zh-CN/model_zoo/recognition/pp-tsm.md)
    - [SlowFast](docs/zh-CN/model_zoo/recognition/slowfast.md)
- 动作定位 [介绍](docs/zh-CN/model_zoo/README.md)
    - [BMN](docs/zh-CN/model_zoo/localization/bmn.md)
- 时空动作检测：
    - Coming Soon!
    
## 技术详解

- **丰富的模型种类**  
    PaddleVideo包含视频分类和动作定位方向的多个主流领先模型，其中TSN, TSM和SlowFast是End-to-End的视频分类模型，Attention LSTM是比较流行的视频特征序列模型，BMN是视频动作定位模型。TSN是基于2D-CNN的经典解决方案，TSM是基于时序移位的简单高效视频时空建模方法，SlowFast是FAIR在ICCV2019提出的3D视频分类模型，特征序列模型Attention LSTM速度快精度高。BMN模型是百度自研模型，为2019年ActivityNet夺冠方案。基于百度飞桨产业实践，我们自研并开源了ppTSM，该模型基于TSM进行优化，在保持模型参数量和计算量不增加的前提下，精度得到大幅提升。同时，我们的通用优化策略可以广泛适用于各种视频模型，未来我们将进行更多的模型优化工作，比如TSN、SlowFast、X3D等，敬请期待。  

- **3000分类预训练模型**  
    飞桨大规模视频分类模型VideoTag基于百度短视频业务千万级数据，支持3000个源于产业实践的实用标签，具有良好的泛化能力，非常适用于国内大规模（千万/亿/十亿级别）短视频分类场景的应用。VideoTag采用两阶段建模方式，即图像建模和序列学习。第一阶段，使用少量视频样本（十万级别）训练大规模视频特征提取模型(Extractor)；第二阶段，使用千万级数据训练预测器(Predictor)，最终实现在超大规模（千万/亿/十亿级别）短视频上产业应用。  

- **SOTA算法PPTSM**  
    多年来，深度学习已经成为视频理解的标准。一个直观的视频理解方法是直接使用2D CNN。然而，二维CNN在单个帧上无法对时间信息进行建模。3D CNNs可以联合学习时空特征，而3D CNN的计算成本较大，使得生产部署成本较高。TSM模型（Accepted by ICCV 2019）有效平衡了计算成本和模型效率，是一个准确和高效的实用视频理解模型，在工业界广泛应用。PaddleVideo基于paddle2.0对TSM进行改进，在不增加参数量和计算量的情况下，在多个数据集上精度显著超过TSM论文精度，比如UCF101、Kinetics-400数据集上分别提升5.5%、3.5%。在仅用ImageNet pretrain情况下，PP-TSM在UCF101和Kinetics400数据集top1分别达到89.5%和73.5%，pp-TSM在Kinetics400上top1精度为73.5%，是至今为止开源的2D视频模型中在相同条件下的最高性能。 

- **更快的训练速度**  
视频任务相比于图像任务的训练往往更加耗时，其原因主要有两点:  
    数据：视频解码耗时。mp4/mkv等视频文件都是经过encode后的压缩文件，通过需要经过解码和抽帧步骤才能得到原始的图像数据流，之后经过图像变换/增强操作才能将其喂入网络进行训练。如果视频帧数多，解码过程是极其耗时的。  
    模型：视频任务使用的模型通常有更大的参数量与计算量。为学习时序特征，视频模型一般会使用3D卷积核/(2+1)D/双流网络，这都会使得模型的参数量与计算量大大增加。
为了加速视频模型的训练，可以分别从模型角度和数据角度考虑。  
模型上可以通过op融合或混合精度训练的方式提升op运算效率。针对TSM模型，我们实现了temporal shift op，在节省显存的同时加速训练过程。  
数据角度，我们采用更优的解码库Decord、融合共享内存等优化的多进程加速Dataloader、纯GPU解码方案DALI、预先解码视频为图像帧加速方案、Multigrid加速策略算法。  
在PaddleVideo中， 8卡v100/32G机器上，SlowFast在训练全量Kinetics-400数据集358个epoch只需要4天；TSM/ppTSM在训练全量Kinetics-400数据集80个epoch只需要2天；均远远快于竞品。  
。

## 许可证书
本项目的发布受[Apache 2.0 license](LICENSE)许可认证。


## 贡献
我们欢迎您的任何贡献并感谢您的支持，更多信息请参考 [contribution guidelines](docs/CONTRIBUTING.md).
