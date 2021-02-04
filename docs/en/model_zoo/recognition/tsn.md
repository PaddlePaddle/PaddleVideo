[简体中文](../../../zh-CN/model_zoo/recognition/tsn.md) | English

# TSN

## Introduction

Temporal Segment Network (TSN) 是视频分类领域经典的基于2D-CNN的解决方案。该方法主要解决视频的长时间行为判断问题，通过稀疏采样视频帧的方式代替稠密采样，既能捕获视频全局信息，也能去除冗余，降低计算量。最终将每帧特征平均融合后得到视频的整体特征，并用于分类。本代码实现的模型为基于单路RGB图像的TSN网络结构，Backbone采用ResNet-50结构。

详细内容请参考ECCV 2016年论文[Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859)

## 数据准备

K400数据下载及准备请参考[数据](../../dataset/K400.md)

UCF101数据下载及准备请参考[数据](../../dataset/ucf101.md)


## 模型训练

- 加载在ImageNet1000上训练好的ResNet50权重作为Backbone初始化参数，请下载此[模型参数](https://paddlemodels.bj.bcebos.com/video_classification/ResNet50_pretrained.tar.gz)并解压，并将路径添加到configs中 BACKBONE字段下
或用-o 参数进行添加，``` -o MODEL.HEAD.pretrained="" ``` 具体参考[conifg](../../config.md)

-下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/TSN.pdparams), 通过`--weights`指定权重存
放路径进行finetune等开发

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

- 指定`--weights`参数，下载已发布模型[model](https://paddlemodels.bj.bcebos.com/video_classification/TSN.pdparams)进行模型测试


当取如下参数时，在Kinetics400的validation数据集下评估精度如下:

| seg\_num | target\_size | Top-1 |
| :------: | :----------: | :----: |
| 3 | 224 | 0.66 |
| 7 | 224 | 0.67 |

## 模型推理

```bash
python3 predict.py --test --weights=
```

## 参考论文

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool
