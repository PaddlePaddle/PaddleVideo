简体中文 | [English](../../en/model_zoo/README.md)

# 前沿算法与模型

## 1. 概要

PaddleVideo包含视频理解方向众多模型，包括基于RGB的行为识别模型，基于骨骼点的行为识别模型、时序动作检测模型、时序分割模型、时空动作检测模型、视频目标分割模型、多模态模型。其中基于RGB的行为识别方向是PaddleVideo核心建设的方向，因其训练得到的好的特征提取器提取的特征，是众多下游任务的基础输入。

与图像识别不同的是，行为识别任务的核心是提取时序信息。按模型结构的不同，基于RGB的行为识别方法大体上可以分为基于2D网络、基于3D网络、基于RNN以及基于Transformer结构的模型。2D网络一般会使用图像预训练模型配合时序模块提取时序信息，比如TSN、TSM等，简单高效。由于视频多一个时序维度，因此很自然的会使用3D卷积提取时序信息，比如I3D、SlowFast。3D模型的计算量一般比较大，训练迭代次数也更多一些。基于RNN的网络以视频特征作为输入，利用RNN提取时序信息，如AttentionLSTM。近期学界涌现了众多基于Transformer结构的行为识别网络，如TimeSformer、VideoSwin。相较于卷积网络，transformer结构的网络精度更高，计算量也会大些。

PaddleVideo自研并开源了PP-TSM，该模型基于TSM进行优化，在保持模型参数量和计算量不增加的前提下，精度得到大幅提升，欢迎使用。更多前沿模型复现与基础模型优化工作，敬请期待～

## 2. 模型概览

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="5" style="font-weight:bold;">行为识别方法</td>
  </tr>
  <tr>
    <td><a href="./recognition/pp-tsm.md">PP-TSM</a> (PP series)</td>
    <td><a href="./recognition/pp-tsn.md">PP-TSN</a> (PP series)</td>
    <td><a href="./recognition/pp-timesformer.md">PP-TimeSformer</a> (PP series)</td>
    <td><a href="./recognition/tsn.md">TSN</a> (2D’)</td>
    <td><a href="./recognition/tsm.md">TSM</a> (2D‘)</td>
  <tr>
    <td><a href="./recognition/slowfast.md">SlowFast</a> (3D’)</td>
    <td><a href="./recognition/timesformer.md">TimeSformer</a> (Transformer‘)</td>
    <td><a href="./recognition/videoswin.md">VideoSwin</a> (Transformer’)</td>
    <td><a href="./recognition/tokenshift_transformer.md">TokenShift</a> (3D’)</td>
    <td><a href="./recognition/attention_lstm.md">AttentionLSTM</a> (RNN‘)</td>
  </tr>
  <tr>
    <td><a href="./recognition/movinet.md">MoViNet</a> (Lite‘)</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">基于骨骼点的行为识别方法</td>
  </tr>
  <tr>
    <td><a href="./recognition/stgcn.md">ST-GCN</a> (GCN’)</td>
    <td><a href="./recognition/agcn.md">AGCN</a> (GCN‘)</td>
    <td><a href="./recognition/agcn2s.md">2s-AGCN</a> (GCN‘)</td>
    <td><a href="./recognition/ctrgcn.md">CTR-GCN</a> (GCN‘)</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">时序动作检测方法</td>
  </tr>
  <tr>
    <td><a href="./localization/bmn.md">BMN</a> (One-stage‘)</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">视频时序分割</td>
  </tr>
  <tr>
    <td><a href="./segmentation/mstcn.md">MS-TCN</a> </td>
    <td><a href="./segmentation/asrf.md">ASRF</a> </td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">时空动作检测方法</td>
  </tr>
  <tr>
    <td><a href="./detection/SlowFast_FasterRCNN.md">SlowFast+Fast R-CNN</a>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">多模态</td>
  </tr>
  <tr>
    <td><a href="./multimodal/actbert.md">ActBERT</a> (Learning‘)</td>
    <td><a href="../../../applications/T2VLAD/README.md">T2VLAD</a> (Retrieval‘)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">视频目标分割</td>
  </tr>
  <tr>
    <td><a href="./segmentation/cfbi.md">CFBI</a> (Semi‘)</td>
    <td><a href="../../../applications/EIVideo/EIVideo/docs/zh-CN/manet.md">MA-Net</a> (Supervised‘)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">单目深度估计</td>
  </tr>
  <tr>
    <td><a href="./estimation/adds.md">ADDS</a> (Unsupervised‘)</td>
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

## 4. Benchmark

各模型训练推理速度参考 [Benchmark](../benchmark.md).
