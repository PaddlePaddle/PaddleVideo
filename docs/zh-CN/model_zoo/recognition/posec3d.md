[English](../../../en/model_zoo/recognition/posec3d.md) | 简体中文

# PoseC3D基于骨骼的行为识别模型

---
## 内容

- [PoseC3D基于骨骼的行为识别模型](#posec3d基于骨骼的行为识别模型)
  - [内容](#内容)
  - [模型简介](#模型简介)
  - [数据准备](#数据准备)
  - [模型训练](#模型训练)
    - [UCF101数据集训练](#ucf101数据集训练)
  - [模型测试](#模型测试)
    - [UCF101数据集模型测试](#ucf101数据集模型测试)
  - [模型推理](#模型推理)
    - [导出inference模型](#导出inference模型)
    - [使用预测引擎推理](#使用预测引擎推理)
  - [参考论文](#参考论文)


## 模型简介


人体骨架作为人类行为的一种简洁的表现形式，近年来受到越来越多的关注。许多基于骨架的动作识别方法都采用了图卷积网络（GCN）来提取人体骨架上的特征。尽管在以前的工作中取得了积极的成果，但基于GCN的方法在健壮性、互操作性和可扩展性方面受到限制。在本文中，作者提出了一种新的基于骨架的动作识别方法PoseC3D，它依赖于3D热图堆栈而不是图形序列作为人体骨架的基本表示。与基于GCN的方法相比，PoseC3D在学习时空特征方面更有效，对姿态估计噪声更具鲁棒性，并且在跨数据集环境下具有更好的通用性。此外，PoseC3D可以在不增加计算成本的情况下处理多人场景，其功能可以在早期融合阶段轻松与其他模式集成，这为进一步提升性能提供了巨大的设计空间。在四个具有挑战性的数据集上，PoseC3D在单独用于Keletons和与RGB模式结合使用时，持续获得优异的性能。

## 数据准备

UCF-101-Skeleton数据集来自mmaction2项目，是由ResNet50作为主干网的Faster-RCNN识别人类，然后使用HRNet-w32实现动作估计。地址如下:

[https://github.com/open-mmlab/mmaction2/tree/master/tools/data/skeleton](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/skeleton)

以及预训练模型下载地址:

[https://aistudio.baidu.com/aistudio/datasetdetail/140593](https://aistudio.baidu.com/aistudio/datasetdetail/140593)

## 模型训练

### UCF101数据集训练

- UCF101数据集使用单卡训练，启动命令如下:

```bash
python3.7 main.py --validate -c configs/recognition/posec3d/posec3d.yaml --weights res3d_k400.pdparams
```



- 您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的，参数用法请参考[config](../../tutorials/config.md)。


## 模型测试

### UCF101数据集模型测试

- 模型测试的启动命令如下：

```bash
python3.7 main.py --test -c configs/recognition/posec3d/posec3d.yaml  -w output/PoseC3D/PoseC3D_epoch_0012.pdparams
```

- 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。


模型在UCF101数据集上baseline实验精度如下:

| Test_Data | Top-1 | checkpoints |
| :----: | :----: | :---- |
| UCF101 test1 | 87.05 | [PoseC3D_ucf101.pdparams]() |



## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/recognition/posec3d/posec3d.yaml \
                                -p data/PoseC3D_ucf101.pdparams \
                                -o inference/PoseC3D
```

上述命令将生成预测所需的模型结构文件`PoseC3D.pdmodel`和模型权重文件`PoseC3D.pdiparams`。

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example_UCF101_skeleton.pkl\
                           --config configs/recognition/posec3d/posec3d.yaml \
                           --model_file inference/PoseC3D/PoseC3D.pdmodel \
                           --params_file inference/PoseC3D/PoseC3D.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

输出示例如下:

```
Current video file: data/example_UCF101_skeleton.pkl
	top-1 class: 0
	top-1 score: 0.6731489896774292
```

可以看到，使用在UCF101上训练好的PoseC3D模型对`data/example_UCF101_skeleton.pkl`进行预测，输出的top1类别id为`0`，置信度为0.67。

## 参考论文

- [Revisiting Skeleton-based Action Recognition](https://arxiv.org/pdf/2104.13586v1.pdf), Haodong Duan, Yue Zhao, Kai Chen, Dian Shao, Dahua Lin, Bo Dai
