简体中文 | [English](../../en/model_zoo/README.md)


# 概要
PaddleVideo包含视频分类和动作定位方向的多个主流领先模型，其中TSN, TSM和SlowFast是End-to-End的视频分类模型，Attention LSTM是比较流行的视频特征序列模型，BMN是视频动作定位模型。TSN是基于2D-CNN的经典解决方案，TSM是基于时序移位的简单高效视频时空建模方法，SlowFast是FAIR在ICCV2019提出的3D视频分类模型，特征序列模型Attention LSTM速度快精度高。BMN模型是百度自研模型，为2019年ActivityNet夺冠方案。基于百度飞桨产业实践，我们自研并开源了ppTSM，该模型基于TSM进行优化，在保持模型参数量和计算量不增加的前提下，精度得到大幅提升。同时，我们的通用优化策略可以广泛适用于各种视频模型，未来我们将进行更多的模型优化工作，比如TPN、SlowFast、X3D等，敬请期待。


## 模型概览

| 领域 | 模型 | 配置 | 测试集 | 精度指标 | 精度% | 下载链接 |
| :--------------- | :--------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| 行为识别 | [**PP-TSM**](./recognition/pp-tsm.md) | [pptsm.yaml](../../../configs/recognition/pptsm/pptsm_k400_frames_dense.yaml) | [Kinetics-400](../dataset/k400.md) | Top-1 | 76.16 | [ppTSM.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense_distill.pdparams) |
| 行为识别| [**PP-TSN**](./recognition/pp-tsn.md) | [pptsn.yaml](../../../configs/recognition/pptsn/pptsn_k400_frames.yaml) | [Kinetics-400](../dataset/k400.md) | Top-1 | 75.06 | [ppTSN.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSN_k400_8.pdparams) |
| 行为识别 | [AGCN](./recognition/agcn.md) | [agcn.yaml](../../../configs/recognition/agcn/agcn_fsd.yaml) | [FSD-10](../dataset/fsd10.md) | Top-1 | 90.66 | AGCN.pdparams |
| 行为识别 | [ST-GCN](./recognition/stgcn.md) | [stgcn.yaml](../../../configs/recognition/stgcn/stgcn_fsd.yaml) | [FSD-10](../dataset/fsd10.md) | Top-1 | 86.66 |  [STGCN.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/STGCN_fsd.pdparams) |
| 行为识别 | [TimeSformer](./recognition/timesformer.md) | [timesformer.yaml](../../../configs/recognition/timesformer/timesformer_k400_videos.yaml) | [Kinetics-400](../dataset/k400.md) | Top-1 | 77.29 | [TimeSformer.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TimeSformer_k400.pdparams) |
| 行为识别 | [SlowFast](./recognition/slowfast.md) | [slowfast_multigrid.yaml](../../../configs/recognition/slowfast/slowfast_multigrid.yaml) | [Kinetics-400](../dataset/k400.md) | Top-1 | 75.84 | [SlowFast.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast_8*8.pdparams) |
| 行为识别 | [TSM](./recognition/tsm.md) | [tsm.yaml](../../../configs/recognition/tsm/tsm_k400_frames.yaml)  | [Kinetics-400](../dataset/k400.md) | Top-1 | 70.86 | [TSM.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/TSM/TSM_k400.pdparams) |
| 行为识别 | [TSN](./recognition/tsn.md) | [tsn.yaml](../../../configs/recognition/tsn/tsn_k400_frames.yaml) | [Kinetics-400](../dataset/k400.md) | Top-1 | 69.81 | [TSN.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TSN_k400.pdparams) |
| 行为识别 | [AttentionLSTM](./recognition/attention_lstm.md) | [attention_lstm.yaml](../../../configs/recognition/attention_lstm/attention_lstm.yaml) | [Youtube-8M](../dataset/youtube8m.md) | Hit@1 | 89.0 | [AttentionLstm.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/AttentionLstm/AttentionLstm.pdparams) |
| 视频动作定位| [BMN](./localization/bmn.md) | [bmn.yaml](../../../configs/localization/bmn.yaml) | [ActivityNet](../dataset/ActivityNet.md) |  AUC | 67.23 | [BMN.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/BMN/BMN.pdparams) |


# 参考文献

- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen

- [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702), Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, Shilei Wen.

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982), Feichtenhofer C, Fan H, Malik J, et al.

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool

- [Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383v1), Ji Lin, Chuang Gan, Song Han

- [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/pdf/2102.05095.pdf) Gedas Bertasius, Heng Wang, Lorenzo Torresani

- [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455), Sijie Yan, Yuanjun Xiong, Dahua Lin

- [Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1805.07694), Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu

- [Skeleton-Based Action Recognition with Multi-Stream Adaptive Graph Convolutional Networks](https://arxiv.org/abs/1912.06971), Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu
