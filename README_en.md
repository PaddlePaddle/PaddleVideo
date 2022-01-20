[简体中文](README.md) | English

# PaddleVideo

​ 💖 **Welcome everyone to scan the code and join the group discussion** 💖
<div align="center">
  <img src="docs/images/user_group.png" width=250/></div>

## Introduction

![python version](https://img.shields.io/badge/python-3.7+-orange.svg) ![paddle version](https://img.shields.io/badge/PaddlePaddle-2.0-blue )


PaddleVideo is a video model development kit produced by [PaddlePaddle Official](https://www.paddlepaddle.org.cn/?fr=paddleEdu_github), which aims to help developers better conduct academic research and industrial practice in the video field.

<div align="center">
  <img src="docs/images/home.gif" width="450px"/><br>
</div>


## Model and Applications

### Model zoo

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="5" style="font-weight:bold;">Action recognition method</td>
  </tr>
  <tr>
    <td><a href="./docs/en/model_zoo/recognition/pp-tsm.md">PP-TSM</a> (PP series)</td>
    <td><a href="./docs/en/model_zoo/recognition/pp-tsn.md">PP-TSN</a> (PP series)</td>
    <td><a href="./docs/en/model_zoo/recognition/pp-timesformer.md">PP-TimeSformer</a> (PP series)</td>
    <td><a href="./docs/en/model_zoo/recognition/tsn.md">TSN</a> (2D’)</td>
    <td><a href="./docs/en/model_zoo/recognition/tsm.md">TSM</a> (2D')</td>
  <tr>
    <td><a href="./docs/en/model_zoo/recognition/slowfast.md">SlowFast</a> (3D’)</td>
    <td><a href="./docs/en/model_zoo/recognition/timesformer.md">TimeSformer</a> (Transformer')</td>
    <td><a href="./docs/en/model_zoo/recognition/videoswin.md">VideoSwin</a> (Transformer’)</td>
    <td><a href="./docs/en/model_zoo/recognition/attention_lstm.md">AttentionLSTM</a> (RNN')</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Skeleton based action recognition</td>
  </tr>
  <tr>
    <td><a href="./docs/en/model_zoo/recognition/stgcn.md">ST-GCN</a> (Custom’)</td>
    <td><a href="./docs/en/model_zoo/recognition/agcn.md">AGCN</a> (Adaptive')</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Sequence action detection method</td>
  </tr>
  <tr>
    <td><a href="./docs/en/model_zoo/localization/bmn.md">BMN</a> (One-stage')</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Spatio-temporal motion detection method</td>
  </tr>
  <tr>
    <td><a href="slowfast.md">SlowFast+Fast R-CNN</a>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Multimodal</td>
  </tr>
  <tr>
    <td><a href="./docs/en/model_zoo/multimodal/actbert.md">ActBERT</a> (Learning')</td>
    <td><a href="">T2VLAD</a> (Retrieval')</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Video target segmentation</td>
  </tr>
  <tr>
    <td><a href="./docs/en/model_zoo/segmentation/cfbi.md">CFBI</a> (Semi')</td>
    <td><a href="./applications/EIVideo/EIVideo/docs/en/manet.md">MA-Net</a> (Supervised')</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Monocular depth estimation</td>
  </tr>
  <tr>
    <td><a href="./docs/en/model_zoo/estimation/adds.md">ADDS</a> (Unsupervised‘)</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

- Please refer to [Installation Instructions](docs/zh-CN/install.md) and [Usage Guide](docs/zh-CN/usage.md) before using the model library.

### Applications

| Applications | Descriptions |
| :--------------- | :------------ |
| [FootballAction]() | Football action detection solution|
| [BasketballAction](applications/BasketballAction) | Basketball action detection solution |
| [TableTennis](applications/ableTennis) | Table tennis action recognition solution|
| [FigureSkating](applications/FigureSkating) | Figure skating action recognition solution|
| [VideoTag](applications/VideoTag) | 3000-category large-scale video classification solution |
| [MultimodalVideoTag](applications/MultimodalVideoTag) | Multimodal video classification solution|
| [VideoQualityAssessment](applications/VideoQualityAssessment) | Video quality assessment solution|
| [PP-Care](applications/PP-Care) | 3DMRI medical image recognition solution |
| [EIVideo](applications/EIVideo) | Interactive video segmentation tool|
| [Anti-UAV](applications/Anti-UAV) |UAV detection solution|

## Documentation tutorial
- AI-Studio Tutorial
    - [[Official] Paddle2.1 realizes video understanding optimization model -- PP-TSM](https://aistudio.baidu.com/aistudio/projectdetail/3399656?contributionType=1)
    - [[Official] Paddle2.1 realizes video understanding optimization model -- PP-TSN](https://aistudio.baidu.com/aistudio/projectdetail/2879980?contributionType=1)
    - [[Official] Paddle 2.1 realizes the classic model of video understanding - TSN](https://aistudio.baidu.com/aistudio/projectdetail/2250682)
    - [[Official] Paddle 2.1 realizes the classic model of video understanding - TSM](https://aistudio.baidu.com/aistudio/projectdetail/2310889)
    - [BMN video action positioning](https://aistudio.baidu.com/aistudio/projectdetail/2250674)
    - [ST-GCN Tutorial for Figure Skate Skeleton Point Action Recognition](https://aistudio.baidu.com/aistudio/projectdetail/2417717)
- Contribute code
    - [How to add a new algorithm](./docs/zh-CN/contribute/add_new_algorithm.md)
    - [Configuration system design analysis](./docs/en/tutorials/config.md)
    - [How to mention PR](./docs/zh-CN/contribute/how_to_contribute.md)


## Tournament Support
- [CCKS 2021: Knowledge Augmented Video Semantic Understanding](https://www.biendata.xyz/competition/ccks_2021_videounde)
