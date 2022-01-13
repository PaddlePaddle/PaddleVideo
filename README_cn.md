[English](README.md) | 中文

# PaddleVideo

## 近期活动

🌟  **1月17号-21号《产业级视频技术与应用案例》** 🌟
- 【1月17号20:15-21:30】视频技术导论及医疗行业典型案例
- 【1月18号20:15-21:30】视频内容智能分析和生产解决方案
- 【1月19号20:15-21:30】体育+安全防范行业中的行为识别
- 【1月20号20:15-21:30】顶会冠军视频分割算法深度解密
- 【1月21号20:15-21:30】多模态学习和检索方法

👀 **报名链接**: https://paddleqiyeban.wjx.cn/vj/QIValIZ.aspx?udsid=419689

​																	  💖 **欢迎大家扫码入群讨论** 💖
<div align="center">
  <img src="docs/images/user_group.png" width=250/></div>

## 简介

![python version](https://img.shields.io/badge/python-3.7+-orange.svg) ![paddle version](https://img.shields.io/badge/PaddlePaddle-2.0-blue)


PaddleVideo是[飞桨官方](https://www.paddlepaddle.org.cn/?fr=paddleEdu_github)出品的视频模型开发套件，旨在帮助开发者更好的进行视频领域的学术研究和产业实践。

<div align="center">
  <img src="docs/images/home.gif" width="450px"/><br>
</div>

### **⭐如果本项目对您有帮助，欢迎点击页面右上方star~ ⭐**


### 模型

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="5" style="font-weight:bold;">行为识别方法</td>
  </tr>
  <tr>
    <td><a href="./docs/zh-CN/model_zoo/recognition/pp-tsm.md">PP-TSM</a> (PP series)</td>
    <td><a href="./docs/zh-CN/model_zoo/recognition/pp-tsn.md">PP-TSN</a> (PP series)</td>
    <td><a href="./docs/zh-CN/model_zoo/recognition/pp-timesformer.md">PP-TimeSformer</a> (PP series)</td>
    <td><a href="./docs/zh-CN/model_zoo/recognition/tsn.md">TSN</a> (2D’)</td>
    <td><a href="./docs/zh-CN/model_zoo/recognition/tsm.md">TSM</a> (2D‘)</td>
  <tr>
    <td><a href="./docs/zh-CN/model_zoo/recognition/slowfast.md">SlowFast</a> (3D’)</td>
    <td><a href="./docs/zh-CN/model_zoo/recognition/timesformer.md">TimeSformer</a> (Transformer‘)</td>
    <td><a href="./docs/zh-CN/model_zoo/recognition/videoswin.md">VideoSwin</a> (Transformer’)</td>
    <td><a href="./docs/zh-CN/model_zoo/recognition/attention_lstm.md">AttentionLSTM</a> (RNN‘)</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">基于骨骼点的动作识别方法</td>
  </tr>
  <tr>
    <td><a href="./docs/zh-CN/model_zoo/recognition/stgcn.md">ST-GCN</a> (Custom’)</td>
    <td><a href="./docs/zh-CN/model_zoo/recognition/agcn.md">AGCN</a> (Adaptive‘)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">时序动作检测方法</td>
  </tr>
  <tr>
    <td><a href="./docs/zh-CN/model_zoo/localization/bmn.md">BMN</a> (One-stage‘)</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">时空动作检测方法</td>
  </tr>
  <tr>
    <td><a href="slowfast.md">SlowFast+Fast R-CNN</a>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">多模态</td>
  </tr>
  <tr>
    <td><a href="./docs/zh-CN/model_zoo/multimodal/actbert.md">ActBERT</a> (Learning‘)</td>
    <td><a href="">T2VLAD</a> (Retrieval‘)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">视频目标分割</td>
  </tr>
  <tr>
    <td><a href="./docs/zh-CN/model_zoo/segmentation/cfbi.md">CFBI</a> (Semi‘)</td>
    <td><a href="./applications/EIVideo/EIVideo/docs/zh-CN/manet.md">MA-Net</a> (Supervised‘)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">单目深度估计</td>
  </tr>
  <tr>
    <td><a href="./docs/zh-CN/model_zoo/estimation/adds.md">ADDS</a> (Unsupervised‘)</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

### 应用案例

| Applications | Descriptions |
| :--------------- | :--------: | 
| [FootballAction]() | 足球动作检测方案|
| [BasketballAction](applications/BasketballAction) | 篮球动作检测方案 |
| [TableTennis](applications/ableTennis) | 乒乓球动作识别方案|
| [FigureSkating](applications/FigureSkating) | 花样滑冰动作识别方案|
| [VideoTag](applications/VideoTag) | 3000类大规模视频分类方案 |
| [MultimodalVideoTag](applications/MultimodalVideoTag) | 多模态视频分类方案|
| [VideoQualityAssessment](applications/VideoQualityAssessment) | 视频质量评估方案|
| [PP-Care](applications/PP-Care) | 3DMRI医疗图像识别方案 |
| [EIVideo](applications/EIVideo) | 视频交互式分割工具|
| [Anti-UAV](applications/Anti-UAV) |无人机检测方案|


## 文档教程
- 免费视频课程、PPT、AIStudio教程（提供免费在线GPU算力）
    - [飞桨视频库全面解析](https://aistudio.baidu.com/aistudio/course/introduce/6742)
    - [视频分类及动作识别介绍](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/tutorials/summarize.md)
    - [【官方】Paddle 2.1实现视频理解经典模型 - TSN](https://aistudio.baidu.com/aistudio/projectdetail/2250682)
    - [【官方】Paddle 2.1实现视频理解经典模型 - TSM](https://aistudio.baidu.com/aistudio/projectdetail/2310889)
    - [BMN视频动作定位](https://aistudio.baidu.com/aistudio/projectdetail/2250674)
    - [花样滑冰选手骨骼点动作识别ST-GCN教程](https://aistudio.baidu.com/aistudio/projectdetail/2417717)
- 快速入门
    - [安装说明](docs/zh-CN/install.md)
    - [使用指南](docs/zh-CN/usage.md)
- 代码组织
    - [模型库设计思路详解](docs/zh-CN/tutorials/modular_design.md)
    - [配置模块参数详解](docs/zh-CN/tutorials/config.md)
- 丰富的模型库
    - [视频分类](docs/zh-CN/model_zoo/README.md)
       - [TSN](docs/zh-CN/model_zoo/recognition/tsn.md)
       - [TSM](docs/zh-CN/model_zoo/recognition/tsm.md)
       - [PP-TSM](docs/zh-CN/model_zoo/recognition/pp-tsm.md)
       - [PP-TSN](docs/zh-CN/model_zoo/recognition/pp-tsn.md)
       - [PP-TimeSformer](docs/zh-CN/model_zoo/recognition/pp-timesformer.md)
       - [VideoSwin](docs/zh-CN/model_zoo/recognition/videoswin.md)
       - [SlowFast](docs/zh-CN/model_zoo/recognition/slowfast.md)
       - [TimeSformer](docs/zh-CN/model_zoo/recognition/timesformer.md)
       - [Attention-LSTM](docs/zh-CN/model_zoo/recognition/attention_lstm.md)
    - [动作定位](docs/zh-CN/model_zoo/README.md)
       - [BMN](docs/zh-CN/model_zoo/localization/bmn.md)
    - [基于骨骼的行为识别](docs/zh-CN/model_zoo/README.md)
       - [ST-GCN](docs/zh-CN/model_zoo/recognition/stgcn.md)
       - [AGCN](docs/zh-CN/model_zoo/recognition/agcn.md)
    - [基于自监督的单目深度估计](docs/zh-CN/model_zoo/README.md)
       - [ADDS](./docs/zh-CN/model_zoo/estimation/adds.md)
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
