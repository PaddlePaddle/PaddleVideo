简体中文 | [English](../../en/model_zoo/README.md)

# 前沿算法与模型

## 1. 概要

PaddleVideo包含视频分类和动作定位方向的多个主流领先模型，其中TSN, TSM和SlowFast是End-to-End的视频分类模型，Attention LSTM是比较流行的视频特征序列模型，BMN是视频动作定位模型，TransNetV2是视频切分模型。TSN是基于2D-CNN的经典解决方案，TSM是基于时序移位的简单高效视频时空建模方法，SlowFast是FAIR在ICCV2019提出的3D视频分类模型，特征序列模型Attention LSTM速度快精度高。BMN模型是百度自研模型，为2019年ActivityNet夺冠方案。基于百度飞桨产业实践，我们自研并开源了PP-TSM，该模型基于TSM进行优化，在保持模型参数量和计算量不增加的前提下，精度得到大幅提升。同时，我们的通用优化策略可以广泛适用于各种视频模型，未来我们将进行更多的模型优化工作，敬请期待。

## 2. 模型概览

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
    <td><a href="./docs/zh-CN/model_zoo/recognition/tokenshift_transformer.md">TokenShift</a> (3D’)</td>
    <td><a href="./docs/zh-CN/model_zoo/recognition/attention_lstm.md">AttentionLSTM</a> (RNN‘)</td>
  </tr>
  <tr>
    <td><a href="./docs/zh-CN/model_zoo/recognition/movinet.md">MoViNet</a> (Lite‘)</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">基于骨骼点的动作识别方法</td>
  </tr>
  <tr>
    <td><a href="./docs/zh-CN/model_zoo/recognition/stgcn.md">ST-GCN</a> (GCN’)</td>
    <td><a href="./docs/zh-CN/model_zoo/recognition/agcn.md">AGCN</a> (GCN‘)</td>
    <td><a href="./docs/zh-CN/model_zoo/recognition/agcn2s.md">2s-AGCN</a> (GCN‘)</td>
    <td><a href="./docs/zh-CN/model_zoo/recognition/ctrgcn.md">CTR-GCN</a> (GCN‘)</td>
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
    <td colspan="5" style="font-weight:bold;">视频时序分割</td>
  </tr>
  <tr>
    <td><a href="./docs/zh-CN/model_zoo/segmentation/mstcn.md">MS-TCN</a> </td>
    <td><a href="./docs/zh-CN/model_zoo/segmentation/asrf.md">ASRF</a> </td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">时空动作检测方法</td>
  </tr>
  <tr>
    <td><a href="docs/zh-CN/model_zoo/detection/SlowFast_FasterRCNN.md">SlowFast+Fast R-CNN</a>
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
    <td><a href="./applications/T2VLAD/README.md">T2VLAD</a> (Retrieval‘)</td>
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


## 3. AI-Studio模型教程

- [【官方】Paddle 2.1实现视频理解优化模型 -- PP-TSM](https://aistudio.baidu.com/aistudio/projectdetail/3399656?contributionType=1)
- [【官方】Paddle 2.1实现视频理解优化模型 -- PP-TSN](https://aistudio.baidu.com/aistudio/projectdetail/2879980?contributionType=1)
- [【官方】Paddle 2.1实现视频理解经典模型 -- TSN](https://aistudio.baidu.com/aistudio/projectdetail/2250682)
- [【官方】Paddle 2.1实现视频理解经典模型 -- TSM](https://aistudio.baidu.com/aistudio/projectdetail/2310889)
- [BMN视频动作定位](https://aistudio.baidu.com/aistudio/projectdetail/2250674)
- [花样滑冰选手骨骼点动作识别ST-GCN教程](https://aistudio.baidu.com/aistudio/projectdetail/2417717)
- [【实践】CV领域的Transformer模型TimeSformer实现视频理解](https://aistudio.baidu.com/aistudio/projectdetail/3413254?contributionType=1)
