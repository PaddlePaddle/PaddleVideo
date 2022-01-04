[简体中文](../../../zh-CN/model_zoo/recognition/cfbi.md) | English

# CFBI

---
## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Test](#Test)
- [Reference](#Reference)

## Introduction

CFBI is a Video Object Segmentation model proposed by Baidu in ECCV 2020. This method consider background should be equally treated and thus propose Collaborative video object segmentation by Foreground-Background Integration (CFBI) approach. Our CFBI implicitly imposes the feature embedding from the target foreground object and its corresponding background to be contrastive, promoting the segmentation results accordingly.  Given the image and target segmentation of the reference frame (the first frame) and the previous frame, the model will predict the segmentation of the current frame.

<div align="center">
<img src="../../../images/cfbi.png" height=400 width=600 hspace='10'/> <br />
</div>


## Data

Please refer to DAVIS data download and preparation doc [DAVIS-data](../../dataset/davis.md)


## Test

- Test scripts:

```bash
python3.7 main.py --test -c configs/segmentation/cfbip_davis.yaml -w CFBIp_davis.pdparams
```

- Predicted results will be saved in `result_root`. To get evaluation metrics, please use [davis2017-evaluation tools](https://github.com/davisvideochallenge/davis2017-evaluation).

Metrics on DAVIS:

| J&F-Mean | J-Mean | J-Recall | J-Decay | F-Mean | F-Recall | F-Decay | checkpoints |
| :------: | :-----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 0.823 | 0.793 | 0.885 | 0.083 | 0.852 | 0.932 | 0.100 | [CFBIp_r101_davis.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/CFBIp_r101_davis.pdparams) |


## Reference

- [Collaborative Video Object Segmentation by Foreground-Background Integration](https://arxiv.org/abs/2003.08333), Zongxin Yang, Yunchao Wei, Yi Yang
