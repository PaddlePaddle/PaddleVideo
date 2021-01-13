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
- [训练加速方案](docs/zh-CN/tutorials/efficiently_training.md)
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

## 许可证书
本项目的发布受[Apache 2.0 license](LICENSE)许可认证。


## 贡献
我们欢迎您的任何贡献并感谢您的支持，更多信息请参考 [contribution guidelines](docs/CONTRIBUTING.md).
