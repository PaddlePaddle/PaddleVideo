[简体中文](../../zh-CN/model_zoo/README.md) | English

# Introduction

We implemented video classification model and action localization model in this repo.

## Model list

| Model | Field  | Description |
| :--------------- | :--------: | :------------: |
| [ppTSM](./recognition/pp-tsm.md) | video recognition| 基于时序移位的简单高效视频时空建模方法 |
| [SlowFast](./recognition/slowfast.md) | video recognition| 3D高精度模型 |
| [TSM](./recognition/tsm.md) | video recognition| 基于时序移位的简单高效视频时空建模方法 |
| [TSN](./recognition/tsn.md) | video recognition| ECCV'16提出的基于2D-CNN经典解决方案 |
| [Attention LSTM](./recognition/attention_lstm.md)  | video recognition| 常用序列模型，速度快精度高 |
| [BMN](./localization/bmn.md) | action localization| 2019年ActivityNet夺冠方案 |


# Reference

- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen

- [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702), Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, Shilei Wen.

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982), Feichtenhofer C, Fan H, Malik J, et al. 

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool

- [Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383v1), Ji Lin, Chuang Gan, Song Han
