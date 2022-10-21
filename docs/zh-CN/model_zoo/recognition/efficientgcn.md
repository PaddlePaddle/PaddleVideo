[English](../../../en/model_zoo/recognition/efficientgcn.md)  | 简体中文

# EfficientGCN基于骨骼的行为识别模型

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)

## 模型简介

![模型结构图](../../../images/efficientgcn.png)

[EfficientGCNv1](https://arxiv.org/pdf/2106.15125v2.pdf)一文提出了基于骨架行为识别的baseline，在论文中，将基于骨架识别的网络分为input branch和 main stream两部分。Input branch 用于提取骨架数据的多模态特征，提取的特征通过concat等操作完成特征融合后将输入main stream中预测动作分类。

## 数据准备

数据下载及处理与ST-GCN一致，详情请参考[NTU-RGBD数据准备](../../dataset/ntu-rgbd.md)

## 模型训练

### NTU-RGBD数据集训练

模型训练参数的配置文件均在`configs/recognition/efficientgcn/`文件夹中，启动命令如下:

```bash
# train cross subject
python main.py --validate -c configs/recognition/efficientgcn/efficientgcn2001.yaml --seed 1
# train cross view
python main.py --validate -c configs/recognition/efficientgcn/efficientgcn2002.yaml --seed 1
```

## 模型测试

### NTU-RGBD数据集模型测试

模型测试参数的配置文件均在`configs/recognition/efficientgcn/`文件夹中，启动命令如下:

```bash
# test cross subject
python main.py --test -c configs/recognition/efficientgcn/efficientgcn2001.yaml -w data/efficientgcn2001.pdparams
# test cross view
python main.py --test -c configs/recognition/efficientgcn/efficientgcn2002.yaml -w data/efficientgcn2002.pdparams
```

* 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。

模型在NTU-RGBD数据集上的测试效果如下

|                |  x-sub  |   x-view   |
| :------------: | :---: | :----: |
| EfficientGCNv1 | 90.2% | 94.9% |

训练日志：[日志](https://github.com/Wuxiao85/paddle_EfficientGCNv/blob/main/workdir/)

模型权重如下：

|      | x-sub                                                        | x-view                                                        |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 地址 | [x-sub](https://github.com/Wuxiao85/paddle_EfficientGCNv/tree/main/pretrain_model/xsub.pdparams) | [x-view](https://github.com/Wuxiao85/paddle_EfficientGCNv/tree/main/pretrain_model/xsub.pdparams)|

## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/recognition/efficientgcn/efficientgcn2001.yaml \
                                -p data/efficientgcn2001.pdparams \
                                -o inference/efficientgcn2001
```

上述命令将生成预测所需的模型结构文件`efficientgcn2001.pdmodel`和模型权重文件`efficientgcn2001.pdiparams`。

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example_NTU-RGB-D_sketeton.npy \
                           --config configs/recognition/efficientgcn/efficientgcn2001.yaml \
                           --model_file inference/efficientgcn2001/efficientgcn2001.pdmodel \
                           --params_file inference/efficientgcn2001/efficientgcn2001.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```


## 参考论文

- [EfficientGCN: Constructing Stronger and Faster Baselines for Skeleton-based Action Recognition ](https://arxiv.org/pdf/2106.15125v2.pdf)
