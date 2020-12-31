[English](README.md) | 中文

# PaddleVideo

## 简介

PaddleVideo 基于全新API设计的PaddlePaddle2.0框架，支持视频领域学术研究和产业实践，为您带来包括视频分类，动作定位，时空动作检测等领域前沿模型。

## 特性

- **模块化设计**
    PaddleVideo 统一了视频理解框架，搭配清晰的配置系统，并将各部分功能拆分到不同组件中进行解耦，可以轻松组合，自定义组件和配置来快速实现视频算法模型。

- **更多的数据集和模型结构**
    PaddleVideo 支持更多的数据集和模型结构，包括Kinectics400，ucf101，YoutTube8M等数据集，模型结构涵盖了视频分类模型TSN，TSM，SlowFast，AttentionLSTM和视频定位模型BMN等。

- **更高指标的模型算法**
    PaddleVideo 提供更高精度的模型结构和解决方案，在基于TSM基础改进的PPTSM上，达到2D网络SOTA效果，Acc1 73.5%相较标准版TSM提升3%且模型参数量持平。

- **更快的解决方案**
    PaddleVideo 提供更快的训练阶段解决方案，在SlowFast模型上提速30%，完整Kinectics-400训练只需要10天。

- **全流程可部署**
    PaddleVideo 提供全流程的预测部署方案，从数据增广到模型设计，从速度测试到模型推理，PaddleVideo方便产出可快速部署的模型，完成同级别业界模型库部署阶段最后一公里。

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
        <ul>
          <li>Recognizer1D</li>
          <li>Recognizer2D</li>
          <li>Recognizer3D</li>
          <li>Localizer</li> 
        </ul>
      </td>
      <td>
        <ul><li><b>Solver</b></li>
          <ul><li><b>Optimizer</b></li>
              <ul>
                <li></li>
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

各模型结构和骨干网络的代表模型在Kinectics-400数据集上Top1精度和单卡Tesla V100上预测速度(VPS)对比图。

<div align="center">
  <img src="docs/images/acc_vps.png" />
</div>

**说明：**
- 图中模型均可在[模型库](#模型库)中获取

## 文档教程

### 入门教程

- [安装说明](docs/zh_CN/install.md)
- [快速开始](docs/zh_CN/getting_started.md)
- [benchmark]()

### 进阶教程
- [模型库整体设计]()
- [配置模块设计]()
- [PPTSM实践]()
- [训练加速方案]()
- [预测部署]()
- [自定义开发]()

### 模型库

- 视频分类 [介绍]()
    - [Attention-LSTM]()
    - [TSN]()
    - [TSM]()
    - [PPTSM]()
    - [SlowFast]()
    - [VideoTag]()
- 动作定位 [介绍]()
    - [BMN]()
- 时空动作检测：
    - Coming Soon!


## 许可证书
本项目的发布受[Apache 2.0 license](LICENSE)许可认证。
