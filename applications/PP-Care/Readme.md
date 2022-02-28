# Video models for 3DMRI

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [实现细节](#实现细节)
- [参考论文](#参考论文)

在开始使用之前，您需要按照以下命令安装额外的依赖包：
```bash
python -m pip install SimpleITK
```

## 模型简介

目前对于医学3D数据如MRI，并无太好的处理手段，大多数2D模型无法获得3D空间层面的特征，而常用的3D模型又需要较大的计算成本。而同时，3D医学数据与常见的视频数据有一定相似之处，我们尝试了通过PaddleVideo中的常见模型解决医学3DMRI数据的分类问题，获得了较好的结果。目前支持PP-TSN、PP-TSM、Slowfast和Timesformer对3DMRI的直接训练。

## 数据准备

数据集包括帕金森患者(PD)与正常(Con)两种类型共378个case，训练集：测试集=300：78，使用数据均为公开数据集，包括*neurocon*, *taowu*, *PPMI*和*OASIS-1*（经过选取），并经过一定格式转换，数据最后的格式均为*name.nii*或*name.nii.gz*，路径与label信息通过txt文件保存，数据集可以通过百度网盘下载：[下载链接](https://pan.baidu.com/s/1eIsHHqnkKNG5x9CGjRONEA?pwd=avug)
- 数据集label格式
```
{
   "0": "Con",
   "1": "PD"
}
```
- 数据集信息文件格式
```
{
   path1 label1
   path2 label2
   ...
}
```
- 数据保存格式
```
{
   |--  datasets
      |--  neurocon
      |--  taowu
      |--  PPMI
      |--  OASIS-1
}
```

## 模型训练

#### 下载并添加预训练模型

1. 对于PP-TSN与PP-TSM，除了可以使用ImageNet1000上训练好的预训练模型（见[PP-TSN预训练模型](../../../docs/zh-CN/model_zoo/recognition/pp-tsn.md)与[PP-TSM预训练模型](../../../docs/zh-CN/model_zoo/recognition/pp-tsm.md))，也可以使用在MRI数据集上预训练的ResNet50权重座位Backbone初始化参数，通过百度网盘下载: [下载链接](https://pan.baidu.com/s/1eIsHHqnkKNG5x9CGjRONEA?pwd=avug)。对于Slowfast与TimeSformer，目前只支持是使用自然数据集的预训练模型，见[Slowfast预训练模型](../../../docs/zh-CN/model_zoo/recognition/slowfast.md)与[Timesformer预训练模型](../../../docs/zh-CN/model_zoo/recognition/timesformer.md)


2. 打开`PaddleVideo/applications/PP-Care/configs/XXX.yaml`，将下载好的权重路径填写到下方`pretrained:`之后，以pptsn_MRI为例

   ```yaml
   MODEL:
       framework: "RecognizerMRI"
       backbone:
           name: "ResNetTSN_MRI"
           pretrained: 将路径填写到此处
   ```

#### 开始训练

- 训练使用显卡数量与输出路径等信息均可以选择，以PP-TSN_MRI的4卡训练为例，训练启动命令如下

  ```bash
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_pptsn_MRI main.py  --validate -c applications/PP-Care/configs/pptsn_MRI.yaml
  ```

## 模型测试

由于各模型均存在随机采样部分，且采样方式存在不同，所以训练日志中记录的验证指标`topk Acc`不代表最终的测试分数，因此在训练完成之后可以用测试模式对最好的模型进行测试获取最终的指标，以PP-TSN_MRI为例，命令如下：

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_pptsn_MRI main.py  --test -c applications/PP-Care/configs/pptsn_MRI.yaml -w "output/ppTSN_MRI/ppTSN_MRI_best.pdparams"
```

当测试配置采用.yaml中参数时，在3DMRI数据的validation数据集上的测试指标如下：

|      backbone      |     head     |  Acc  |
| :----------------: | :----------: | :---: |
|      ResNet50      |    PP-TSN    | 91.07 |
|      ResNet50      |    PP-TSM    | 90.83 |
|     3DResNet50     |   Slowfast   | 91.07 |
| Vision Transformer |  Timesformer | 88.33 |

训练好的模型可以通过百度网盘下载：[下载链接](https://pan.baidu.com/s/1eIsHHqnkKNG5x9CGjRONEA?pwd=avug)


## 模型优化
在实际使用中，可以尝试模型优化策略
- 可以根据MRI数据分布，调整采样率
- 本模型目前未加入过多的数据预处理策略，针对不同数据特性，在本模型基础上加入一定的预处理手段可能会使结果继续提升
- 由于数据量与任务难度限制，本模型目前在准确率上的表现与3DResNet并无显著区别，但对于时间与空间的需求均远小于3D模型


## 参考论文

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/pdf/1608.00859.pdf), Limin Wang, Yuanjun Xiong, Zhe Wang
- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), Geoffrey Hinton, Oriol Vinyals, Jeff Dean
- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982), Feichtenhofer C, Fan H, Malik J, et al.
- [A Multigrid Method for Efficiently Training Video Models](https://arxiv.org/abs/1912.00998), Chao-Yuan Wu, Ross Girshick, et al.
- [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/pdf/2102.05095.pdf), Gedas Bertasius, Heng Wang, Lorenzo Torresani
