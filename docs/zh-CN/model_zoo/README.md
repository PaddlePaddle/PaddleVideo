简体中文 | [English](../../en/model_zoo/README.md)


# 概要
该repo包含视频分类和动作定位方向的多个主流领先模型，其中Attention LSTM是比较流行的视频特征序列模型，TSN, TSM和SlowFast是End-to-End的视频分类模型。Attention LSTM模型速度快精度高，TSN是基于2D-CNN的经典解决方案，TSM是基于时序移位的简单高效视频时空建模方法，SlowFast是FAIR在ICCV2019提出的3D视频分类模型。 BMN模型是百度自研模型，2019年ActivityNet夺冠方案。


## 视频分类模型

[Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen

[Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool

[Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383v1), Ji Lin, Chuang Gan, Song Han

[SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)

[pptsm](recognition/pp-tsm.md)

## 动作定位模型
[BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702), Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, Shilei Wen.
