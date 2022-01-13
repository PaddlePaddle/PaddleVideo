[简体中文](README_cn.md) | English

# PaddleVideo

## recent activities

🌟 **January 17-21 "Industrial Video Technology and Application Cases"** 🌟
- [January 17, 20:15-21:30] Introduction to Video Technology and Typical Cases in the Medical Industry
- [20:15-21:30 on January 18] Video content intelligent analysis and production solutions
- [January 19, 20:15-21:30] Behavior recognition in the sports + security industry
- [20:15-21:30 on January 20] Depth decryption of the video segmentation algorithm of the champion of the top meeting
- [January 21st 20:15-21:30] Multimodal Learning and Retrieval Methods

👀 **Registration link**: https://paddleqiyeban.wjx.cn/vj/QIValIZ.aspx?udsid=419689

​ 💖 **Welcome everyone to scan the code and join the group discussion** 💖
<div align="center">
  <img src="docs/images/user_group.png" width=250/></div>

## Introduction

![python version](https://img.shields.io/badge/python-3.7+-orange.svg) ![paddle version](https://img.shields.io/badge/PaddlePaddle-2.0-blue )


PaddleVideo is a video model development kit produced by [PaddlePaddle Official](https://www.paddlepaddle.org.cn/?fr=paddleEdu_github), which aims to help developers better conduct academic research and industrial practice in the video field.

<div align="center">
  <img src="docs/images/home.gif" width="450px"/><br>
</div>

### **⭐If this project is helpful to you, please click star at the top right of the page~ ⭐**


### Project

| Applications | Descriptions |
| :--------------- | :-------- |
| [FootballAction]() | Football action detection model|
| [BasketballAction](applications/BasketballAction) | Basketball action detection model|
| [TableTennis](applications/TableTennis) | TableTennis action recognition model|
| [FigureSkating](applications/FigureSkating) | FigureSkating action recognition model|
| [VideoTag](applications/VideoTag) | 3k Large-Scale video classification model |
| [MultimodalVideoTag](applications/MultimodalVideoTag) | Multimodal video classification model|
| [VideoQualityAssessment](applications/VideoQualityAssessment) | Video Quality Assessment model|
| [PP-Care](applications/PP-Care) | Video models for 3DMRI |
| [EIVideo](applications/EIVideo) | Efficient interactive video object segmentation tools|
| [Anti-UAV](applications/Anti-UAV) |UAV detection model |

## Featured application effect
- [Featured Application 01: Large-scale Video 3k Class Tag Scheme VideoTag](https://github.com/PaddlePaddle/PaddleVideo/tree/application/VideoTag)

<div align="center">
  <img src="docs/images/VideoTag.gif" width="450px"/><br>
</div>

- [Featured Application 02: Football Action Positioning Solution FootballAction](https://github.com/PaddlePaddle/PaddleVideo/tree/application/FootballAction)

<div align="center">
  <img src="docs/images/FootballAction.gif" width="450px"/><br>
</div>


## Documentation tutorial
- Free video courses, PPT, AIStudio tutorials (free online GPU computing power)
    - [Comprehensive Analysis of Flying Paddle Video Library](https://aistudio.baidu.com/aistudio/course/introduce/6742)
    - [Introduction to Video Classification and Action Recognition](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/tutorials/summarize.md)
    - [[Official] Paddle 2.1 realizes the classic model of video understanding - TSN](https://aistudio.baidu.com/aistudio/projectdetail/2250682)
    - [[Official] Paddle 2.1 realizes the classic model of video understanding - TSM](https://aistudio.baidu.com/aistudio/projectdetail/2310889)
    - [BMN video action positioning](https://aistudio.baidu.com/aistudio/projectdetail/2250674)
    - [ST-GCN Tutorial for Figure Skate Skeleton Point Action Recognition](https://aistudio.baidu.com/aistudio/projectdetail/2417717)
- Quick start
    - [Installation Instructions](docs/zh-CN/install.md)
    - [Usage Guide](docs/zh-CN/usage.md)
- Code organization
    - [Detailed explanation of model library design ideas](docs/zh-CN/tutorials/modular_design.md)
    - [Detailed explanation of configuration module parameters](docs/zh-CN/tutorials/config.md)
- Rich model library
    - [Video classification](docs/zh-CN/model_zoo/README.md)
       - [TSN](docs/en-US/model_zoo/recognition/tsn.md)
       - [TSM](docs/en-US/model_zoo/recognition/tsm.md)
       - [PP-TSM](docs/zh-CN/model_zoo/recognition/pp-tsm.md)
       - [PP-TSN](docs/zh-CN/model_zoo/recognition/pp-tsn.md)
       - [PP-TimeSformer](docs/en-US/model_zoo/recognition/pp-timesformer.md)
       - [VideoSwin](docs/en-US/model_zoo/recognition/videoswin.md)
       - [SlowFast](docs/zh-CN/model_zoo/recognition/slowfast.md)
       - [TimeSformer](docs/en-US/model_zoo/recognition/timesformer.md)
       - [Attention-LSTM](docs/zh-CN/model_zoo/recognition/attention_lstm.md)
    - [Action Positioning](docs/zh-CN/model_zoo/README.md)
       - [BMN](docs/zh-CN/model_zoo/localization/bmn.md)
    - [Bone-Based Behavior Recognition](docs/zh-CN/model_zoo/README.md)
       - [ST-GCN](docs/zh-CN/model_zoo/recognition/stgcn.md)
       - [AGCN](docs/zh-CN/model_zoo/recognition/agcn.md)
    - [Self-supervised Monocular Depth Estimation](docs/zh-CN/model_zoo/README.md)
       - [ADDS](./docs/en-US/model_zoo/estimation/adds.md)
    - Time and space motion detection <sup>coming soon</sup>
    - ActBERT: Self-Supervised Multimodal Video Text Learning <sup>coming soon</sup>
- Project combat
    - [PP-TSM Practice](docs/zh-CN/tutorials/pp-tsm.md)
    - [Training Accelerate](docs/zh-CN/tutorials/accelerate.md)
    - [Predicted Deployment](docs/en-US/tutorials/deployment.md)
- Auxiliary tools
    - [benchmark](docs/zh-CN/benchmark.md)
    - [Tools](docs/zh-CN/tools.md)
- [Technical Exchange Group](docs/images/user_group.png)
- [Event Support](#Tournament_Support)
- [license](#License)
- [pr](docs/zh-CN/contribute/how_to_contribute.md)


## Tournament_Support
- [CCKS 2021: Knowledge-Enhanced Video Semantic Understanding](https://www.biendata.xyz/competition/ccks_2021_videounderstanding/)
- [Recognition of skeletal points of figure skaters based on flying paddles](https://aistudio.baidu.com/aistudio/competition/detail/115/0/introduction)

## License
This project is released under the [Apache 2.0 license](LICENSE) license.


## Contributions welcome
We welcome any contributions and appreciate your support, see the [contribution guidelines](docs/CONTRIBUTING.md) for more information.

- Many thanks to [mohui37](https://github.com/mohui37), [zephyr-fun](https://github.com/zephyr-fun), [voipchina](https://github.com/voipchina ) contribute relevant code
