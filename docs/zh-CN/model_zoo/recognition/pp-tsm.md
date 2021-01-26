简体中文 | [English](../../../en/model_zoo/recognition/pp-tsm.md)

# PPTSM视频分类模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介

我们基于飞桨框架2.0版本对[TSM模型](./tsm.md)进行了改进，提出了**PPTSM**高精度2D实用视频分类模型。在不增加参数量和计算量的情况下，在UCF-101、Kinetics-400等数据集上精度显著超过原文。模型优化解析请参考[**pptsm实用视频模型优化解析**](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/tutorials/pp-tsm.md)。

<p align="center">
<img src="https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/images/acc_vps.jpeg" height=400 width=650 hspace='10'/> <br />
PPTSM improvement
</p>


## 数据准备

K400数据下载及准备请参考[Kinetics-400数据准备](../../dataset/k400.md)

UCF101数据下载及准备请参考[UCF-101数据准备](../../dataset/ucf101.md)


## 模型训练

### 预训练模型下载

下载图像蒸馏预训练模型[ResNet50_vd_ssld_v2](https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams)作为Backbone初始化参数，或是通过命令行下载

```bash
wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams
```

并将文件路径添加到配置文件中的`MODEL.framework.backbone.pretrained`字段，如下：

```yaml
MODEL:
framework: "Recognizer2D"
    backbone:
        name: "ResNet"
        pretrained: 将路径填写到此处
```

### 开始训练

不同数据格式，不同数据集的训练命令分别如下:

(1) UCF-101 frames格式训练
```bash
python -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/tsm/pptsm.yaml
```

- 模型及训练参数配置请参考配置文件```configs/recognition/tsm/pptsm.yaml```，您可以自定义修改参数配置。


(2) UCF-101 video格式训练

(3) Kinetics-400 frames格式训练

(4) Kinetics-400 video格式训练

- 通过`--weights`指定权重存放路径可进行模型finetune。 `--weights` 参数用法请参考[config](../../tutorials/config.md)



## 模型测试

```bash
python3 main.py --test --weights=""
```

- 指定`--weights`参数，下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/PPTSM.pdparams) 进行模型测试


当取如下参数时，在Kinetics400的验证集下评估精度如下:

| seg\_num | target\_size | Top-1 |
| :------: | :----------: | :----: |
| 8 | 224 | 0.735 |

UCF101验证集(split1)上的评估精度如下：

| seg\_num | target\_size | Top-1 |
| :------: | :----------: | :----: |
| 8 | 224 | 0.8997 |

## 模型推理

```bash
python3 predict.py --test --weights=
```

## 参考论文

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
