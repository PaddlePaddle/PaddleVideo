[简体中文](README_cn.md) | English

# Ma-Net

## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)




## Introduction

This is the paddle implementation of the CVPR2020 paper "[Memory aggregation networks for efficient interactive video object segmentation](https://arxiv.org/abs/2003.13246)".

![avatar](images/1836-teaser.gif)

This code currently supports model test and model training on DAVIS  dataset,  and model inference on any given video will be provided in few days.



## Data

Please refer to DAVIS data download and preparation doc [DAVIS-data](dataloaders/DAVIS2017.md)

## Train and Test
- You can download [pertained model for stage1](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/DeeplabV3_coco.pdparams) decompress it for stage1 training。
  
- You can download [trained model of stage1](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/MaNet_davis2017_stage1.pdparams) decompress it for stage2 training directly skipping stage1 training。
  
```
sh run_local.sh
```

- You can download [our model](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/MANet_davis2017.pdparams) decompress it for testing.



Test accuracy in DAVIS2017:

| J@60  |  AUC  |
| :---: | :---: |
| 0.761 | 0.749 |
