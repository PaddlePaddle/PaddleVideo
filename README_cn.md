[English](README.md) | 中文

# PaddleVideo

## 简介

PaddleVideo 基于全新API设计的PaddlePaddle2.0框架，支持视频领域学术研究和产业实践，为您带来包括视频分类，动作定位，时空动作检测等领域前沿模型。

## 特性

- **模块化设计**
    PaddleVideo 统一了视频理解框架，搭配清晰的配置系统，将各部分功能拆分到不同组件中进行解耦。可以轻松组合，自定义组件和配置来快速实现视频算法模型。

- **更多的数据集和模型结构**
    PaddleVideo 支持更多的数据集和模型结构，包括Kinectics400，ucf101，YoutTube8M等数据集，模型结构涵盖了视频分类模型TSN，TSM，SlowFast，AttentionLSTM和视频定位模型BMN等。

- **更高指标的模型算法**
    PaddleVideo 提供更高精度的模型结构和解决方案，在基于TSM标准版改进的PPTSM上，达到2D网络SOTA效果，Top1 Acc 73.5% 相较标准版TSM提升3%且模型参数量持平。

- **更快的解决方案**
    PaddleVideo 提供更快的训练阶段解决方案，在SlowFast上标准版相较pytorch提速100%，完整训练Kinectics400数据集只需10天。

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
- 图中红色描述的模型均可在[模型库](#模型库) <sup>coming soon</sup>中获取

## 文档教程

### 入门教程

- [安装说明](docs/zh-CN/install.md)
- [快速开始](docs/zh-CN/getting_started.md)
- [benchmark]()  <sup>coming soon</sup>

### 进阶教程
- [模型库整体设计]() <sup>coming soon</sup>
- [配置模块设计]() <sup>coming soon</sup>
- [PPTSM实践]() <sup>coming soon</sup>
- [训练加速方案]() <sup>coming soon</sup>
- [预测部署]() <sup>coming soon</sup>
- [自定义开发]() <sup>coming soon</sup>

### 模型库

- 视频分类 [介绍]() <sup>coming soon</sup>
    - [Attention-LSTM]() <sup>coming soon</sup>
    - [TSN]() <sup>coming soon</sup>
    - [TSM]() <sup>coming soon</sup>
    - [PPTSM]() <sup>coming soon</sup>
    - [SlowFast]() <sup>coming soon</sup>
    - [VideoTag]() <sup>coming soon</sup>
- 动作定位 [介绍]() <sup>coming soon</sup>
    - [BMN]() <sup>coming soon</sup>
- 时空动作检测：
    - Coming Soon!


## 许可证书
本项目的发布受[Apache 2.0 license](LICENSE)许可认证。


## 贡献
我们欢迎您的任何贡献并感谢您的支持，更多信息请参考 [contribution guidelines](docs/CONTRIBUTING.md).
