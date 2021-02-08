[简体中文](../../../zh-CN/model_zoo/recognition/tsm.md) | English

# TSM

---
## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Reference](#Reference)


## Introduction

Temporal Shift Module (TSM) is a popular model that attracts more attention at present. 
The method of moving through channels greatly improves the utilization ability of temporal information without increasing any additional number of parameters and calculation amount. 
Moreover, due to its lightweight and efficient characteristics, it is very suitable for industrial landing.

<div align="center">
<img src="../../../images/tsm_architecture.png" height=250 width=700 hspace='10'/> <br />
</div>

This code implemented single RGB stream of TSM networks. Backbone is ResNet-50.

Please refer to the ICCV 2019 paper for details [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf)

## Data

Please refer to K400 data download and preparation [k400 data preparation](../../dataset/K400.md)

Please refer to UCF101 data download and preparation [ucf101 data preparation](../../dataset/ucf101.md)


## Train

- The parameters of the model are initialized by loading the weights of Resnet50 trained on ImageNet1000. You can download it by yourself 
[Pretrained](https://paddlemodels.bj.bcebos.com/video_classification/ResNet50_pretrained.tar.gz) . Then, you need unzip it and add its path to the filed of `BACKBONE`,which is in the `configs/tsm.yaml`,of course,
you can also use the parameter `-o MODEL.HEAD.pretrained=""`,details can be found in [conifg](../../config.md)

- Download the published model [model](https://paddlemodels.bj.bcebos.com/video_classification/TSM.pdparams), then, you can use '--weights' to specify the weight path for finetune and other development

K400 video training

K400 frames training

UCF101 video training

UCF101 frames training

## Implementation Detail

**data preparation：** The `MP4` data from the Kinetics 400 dataset were read in the model，Segment 'seg_num' is extracted from each piece of data, and 1 frame of image is extracted from each piece of data. After random enhancement of each frame of image, the image is scaled to 'target_size'.

**training strategies：**

*  Momentum optimization algorithm is used for training, and Momentum =0.9
*  L2_decay weight is set to be 1e-4
*  The learning rate decays by  a factor of 10 in the 1/3 and 2/3 of the total epochs

**Parameters Initialization**

****

## Test

```bash
python3 main.py --test --weights=""
```

- Download the published model [model](https://paddlemodels.bj.bcebos.com/video_classification/TSM.pdparams) , then , you need to set the `--weights` for model testing


When the following parameters, the accuracy is evaluated as follows in the Validation dataset of Kinetics400:

| seg\_num | target\_size | Top-1 |
| :------: | :----------: | :----: |
| 8 | 224 | 0.70 |

## Inference

```bash
python3 predict.py --test --weights=
```

## Reference

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
