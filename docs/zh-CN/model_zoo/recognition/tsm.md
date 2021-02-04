[English](../../../en/model_zoo/recognition/tsm.md) | 简体中文

# TSM

## 模型介绍

Temporal Shift Module (TSM) 是当前比较受关注的模型，通过通道移动的方法在不增加任何额外参数量和计算量的情况下极大的提升了模型对于视频时间信息的利用能力，并且由于其具有轻量高效的特点，十分适合工业落地。

<div align="center">
  <img src="../../../images/tsm_architecture.png" >
</div>

本代码实现的模型为基于单路RGB图像的TSM网络结构，Backbone采用ResNet-50结构。

详细内容请参考ICCV 2019年论文 [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf)

## 数据准备

K400数据下载及准备请参考[数据](../../dataset/K400.md)

UCF101数据下载及准备请参考[数据](../../dataset/ucf101.md)


## 模型训练

- 加载在ImageNet1000上训练好的ResNet50权重作为Backbone初始化参数，请下载此[模型参数](https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams),
或是通过命令行下载

```bash
wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams
```

并将路径添加到configs中backbone字段下

```yaml
MODEL:
    framework: "Recognizer2D"
    backbone:
        name: "ResNet"
        pretrained: 将路径填写到此处
```

或用`-o` 参数在```run.sh```或命令行中进行添加
``` -o MODEL.backbone.pretrained="" ```
`-o` 参数用法请参考[conifg](../../config.md)

- 如若进行`finetune`或者[模型测试](#模型测试)等，请下载PaddleVideo的已发布模型[model]()<sup>coming soon</sup>, 通过`--weights`指定权重存放路径。 `--weights` 参数用法请参考[config](../../config.md)

K400 video格式训练

K400 frames格式训练

UCF101 video格式训练

UCF101 frames格式训练

## 实现细节

**数据处理：** 模型读取Kinetics-400数据集中的`mp4`数据，每条数据抽取`seg_num`段，每段抽取1帧图像，对每帧图像做随机增强后，缩放至`target_size`。

**训练策略：**

*  采用Momentum优化算法训练，momentum=0.9
*  l2_decay权重衰减系数为1e-4
*  学习率在训练的总epoch数的1/3和2/3时分别做0.1倍的衰减

**参数初始化**

## 模型测试

```bash
python3 main.py --test --weights=""
```

- 指定`--weights`参数，下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/TSM.pdparams) 进行模型测试


当取如下参数时，在Kinetics400的validation数据集下评估精度如下:

| seg\_num | target\_size | Top-1 |
| :------: | :----------: | :----: |
| 8 | 224 | 0.70 |

## 模型推理

```bash
python3 predict.py --test --weights=
```

## 参考论文

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
