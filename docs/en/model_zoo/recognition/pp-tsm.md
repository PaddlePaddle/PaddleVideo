[简体中文](../../../zh-CN/model_zoo/recognition/pp-tsm.md) | English

# PPTSM

## Introduction

We improve the TSM based on PADDLE 2.0 and propose a optimized model called **PPTSM**. Without increasing the number of parameters and more cost, the accuracy of TSM was significantly improved in UCF101 and KINETIC-400 datasets. Please refer to[《pptsm实用视频模型优化解析》]() <sup>coming soon</sup> for more details.

## Data Preparation

Please refer to K400 data download and preparation doc [data](../../dataset/K400.md)

Please refer to UCF101 data download and preparation doc [data](../../dataset/ucf101.md)


## Training

- The parameters of the model are initialized by loading the weights of Resnet50 trained on ImageNet1000. You can download it by yourself 
[Pretrained]() . Then, you need unzip it and add its path to the filed of `BACKBONE`,which is in the `configs/tsm.yaml`,of course,
you can also use the parameter `-o MODEL.HEAD.pretrained=""`,details can be found in [conifg](../../config.md)

- Download the published model [model](https://paddlemodels.bj.bcebos.com/video_classification/PPTSM.pdparams), then, you can use '--weights' to specify the weight path for finetune and other development


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

- Download the published model [model](https://paddlemodels.bj.bcebos.com/video_classification/PPTSM.pdparams) , then , you need to set the `--weights` for model testing


When the following parameters, the accuracy is evaluated as follows in the Validation dataset of Kinetics400:

| seg\_num | target\_size | Top-1 |
| :------: | :----------: | :----: |
| 8 | 224 | 0.735 |

The accuracy of the evaluation on the UCF101 validation set (Split1) is as follows：

| seg\_num | target\_size | Top-1 |
| :------: | :----------: | :----: |
| 8 | 224 | 0.8997 |

## Inference

```bash
python3 predict.py --test --weights=
```

## Reference

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han