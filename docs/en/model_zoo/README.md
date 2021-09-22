[简体中文](../../zh-CN/model_zoo/README.md) | English

# Introduction

We implemented action recgonition model and action localization model in this repo.

## Model list

| Field | Model | Config | Dataset | Metrics | ACC% | Download |
| :--------------- | :--------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| action recognition | [**PP-TSM**](./recognition/pp-tsm.md) | [pptsm.yaml](../../../configs/recognition/pptsm/pptsm_k400_frames_dense.yaml) | [Kinetics-400](../dataset/k400.md) | Top-1 | 76.16 | [ppTSM.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense_distill.pdparams) |
| action recognition | [**PP-TSN**](./recognition/pp-tsn.md) | [pptsn.yaml](../../../configs/recognition/pptsn/pptsn_k400_videos.yaml) | [Kinetics-400](../dataset/k400.md) | Top-1 | 75.06 | [ppTSN.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSN_k400_8.pdparams) |
| action recognition | [AGCN](./recognition/agcn.md) | [agcn.yaml](../../../configs/recognition/agcn/agcn_fsd.yaml) | [FSD-10](../dataset/fsd10.md) | Top-1 | 90.66 | AGCN.pdparams |
| action recognition | [ST-GCN](./recognition/stgcn.md) | [stgcn.yaml](../../../configs/recognition/stgcn/stgcn_fsd.yaml) | [FSD-10](../dataset/fsd10.md) | Top-1 | 86.66 |  [STGCN.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/STGCN_fsd.pdparams) |
| action recognition | [TimeSformer](./recognition/timesformer.md) | [timesformer.yaml](../../../configs/recognition/timesformer/timesformer_k400_videos.yaml) | [Kinetics-400](../dataset/k400.md) | Top-1 | 77.29 | [TimeSformer.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TimeSformer_k400.pdparams) |
| action recognition | [SlowFast](./recognition/slowfast.md) | [slowfast_multigrid.yaml](../../../configs/recognition/slowfast/slowfast_multigrid.yaml) | [Kinetics-400](../dataset/k400.md) | Top-1 | 75.84 | [SlowFast.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast_8*8.pdparams) |
| action recognition | [TSM](./recognition/tsm.md) | [tsm.yaml](../../../configs/recognition/tsm/tsm_k400_frames.yaml)  | [Kinetics-400](../dataset/k400.md) | Top-1 | 70.86 | [TSM.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/TSM/TSM_k400.pdparams) |
| action recognition | [TSN](./recognition/tsn.md) | [tsn.yaml](../../../configs/recognition/tsn/tsn_k400_frames.yaml) | [Kinetics-400](../dataset/k400.md) | Top-1 | 69.81 | [TSN.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TSN_k400.pdparams) |
| action recognition | [AttentionLSTM](./recognition/attention_lstm.md) | [attention_lstm.yaml](../../../configs/recognition/attention_lstm/attention_lstm.yaml) | [Youtube-8M](../dataset/youtube8m.md) | Hit@1 | 89.0 | [AttentionLstm.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/AttentionLstm/AttentionLstm.pdparams) |
| action detection| [BMN](./localization/bmn.md) | [bmn.yaml](../../../configs/localization/bmn.yaml) | [ActivityNet](../dataset/ActivityNet.md) |  AUC | 67.23 | [BMN.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/BMN/BMN.pdparams) |


# Reference

- [Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification](https://arxiv.org/abs/1711.09550), Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen

- [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702), Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, Shilei Wen.

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982), Feichtenhofer C, Fan H, Malik J, et al.

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool

- [Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383v1), Ji Lin, Chuang Gan, Song Han

- [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/pdf/2102.05095.pdf) Gedas Bertasius, Heng Wang, Lorenzo Torresani
