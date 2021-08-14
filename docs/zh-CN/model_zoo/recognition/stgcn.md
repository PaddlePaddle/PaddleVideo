[English](../../../en/model_zoo/recognition/stgcn.md) | 简体中文

# ST-GCN基于骨骼的行为识别模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介

ST-GCN是AAAI 2018提出的经典的基于骨骼的行为识别模型，通过将图卷积应用在具有拓扑结构的人体骨骼数据上，使用时空图卷积提取时空特征，进行行为识别，极大地提升了基于骨骼的行为识别任务精度。

我们对ST-GCN模型进行了优化，并实现了精度更优的PP-AGCN模型，详见[PP-AGCN基于骨骼的行为识别模型]().


## 数据准备

FSD-10数据下载及准备请参考[FSD-10数据准备](../../dataset/fsd10.md)

## 模型训练

### FSD-10数据集训练

- FSD-10数据集使用单卡训练，启动命令如下:

```bash
python3.7 main.py -c configs/recognition/stgcn/stgcn_fsd.yaml
```

- 由于赛事未提供验证集数据，因此训练时不做valid。

- 您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的，参数用法请参考[config](../../tutorials/config.md)。


## 模型测试

- 训练完成后，模型测试的启动命令如下：

```bash
python3.7 main.py --test -c configs/recognition/stgcn/stgcn_fsd.yaml -w output/STGCN/STGCN_epoch_00060.pdparams
```

- 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。

- 评估结果保存在submission.csv文件中，可在[评测官网]()提交查看得分。

模型在FSD-10数据集上baseline实验精度如下:

| Top-1 | checkpoints |
| :----: | :---- |
| 70.0 | [STGCN_fsd.pdparams]() |


## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/recognition/stgcn/stgcn_fsd.yaml \
                                -p output/STGCN/STGCN_epoch_00060.pdparams \
                                -o inference/STGCN
```

上述命令将生成预测所需的模型结构文件`STGCN.pdmodel`和模型权重文件`STGCN.pdiparams`。

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/fsd10/example_skeleton.npy \
                           --config configs/recognition/stgcn/stgcn_fsd.yaml \
                           --model_file inference/STGCN/STGCN.pdmodel \
                           --params_file inference/STGCN/STGCN.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

输出示例如下:

```
Current video file: data/fsd10/example_skeleton.npy
        top-1 class: 0
        top-1 score: 0.9993312358856201
```

可以看到，使用在FSD-10上训练好的ST-GCN模型对`data/example_skeleton.npy`进行预测，输出的top1类别id为`0`，置信度为0.99。

## 参考论文

- [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455), Sijie Yan, Yuanjun Xiong, Dahua Lin
