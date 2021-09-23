[English](../../../en/model_zoo/recognition/agcn.md) | 简体中文

# AGCN基于骨骼的行为识别模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介


我们对[ST-GCN模型](./stgcn.md)进行了优化，实现了精度更高的AGCN模型，模型优化细节参考[AGCN模型详解]().


## 数据准备

FSD-10数据下载及准备请参考[FSD-10数据准备](../../dataset/fsd10.md)

NTU-RGBD数据下载及准备请参考[NTU-RGBD数据准备](../../dataset/ntu-rgbd.md)

## 模型训练

### FSD-10数据集训练

- FSD-10数据集使用单卡训练，启动命令如下:

```bash
python3.7 main.py -c configs/recognition/agcn/agcn_fsd.yaml
```

- 由于赛事未提供验证集数据，因此训练时不做valid。

- 您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的，参数用法请参考[config](../../tutorials/config.md)。

### NTU-RGBD数据集训练

- NTU-RGBD数据集使用4卡训练，启动命令如下:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_agcn  main.py  --validate -c configs/recognition/agcn/agcn_ntucs.yaml
```

- `agcn_ntucs.yaml`配置文件为NTU-RGB+D数据集按cross-subject划分方式对应的训练配置。


## 模型测试

### FSD-10数据集模型测试

- 模型测试的启动命令如下：

```bash
python3.7 main.py --test -c configs/recognition/agcn/agcn_fsd.yaml  -w output/AGCN/AGCN_epoch_00100.pdparams
```

- 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。

- 评估结果保存在submission.csv文件中，可在[评测官网](https://aistudio.baidu.com/aistudio/competition/detail/115)提交查看得分。

模型在FSD-10数据集上baseline实验精度如下:

| Test_Data | Top-1 | checkpoints |
| :----: | :----: | :---- |
| Test_A | 90.66 | AGCN_fsd.pdparams |


### NTU-RGB+D数据集模型测试

- 模型测试的启动命令如下：

```bash
python3.7 main.py --test -c configs/recognition/agcn/agcn_ntucs.yaml -w output/AGCN/AGCN_best.pdparams
```

- 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。

模型在NTU-RGB+D数据集上实验精度如下:

| split | Top-1 | checkpoints |
| :----: | :----: | :---- |
| cross-subject | 83.27 | AGCN_ntucs.pdparams |


## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/recognition/agcn/agcn_fsd.yaml \
                                -p data/AGCN_fsd.pdparams \
                                -o inference/AGCN
```

上述命令将生成预测所需的模型结构文件`AGCN.pdmodel`和模型权重文件`AGCN.pdiparams`。

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/fsd10/example_skeleton.npy \
                           --config configs/recognition/agcn/agcn_fsd.yaml \
                           --model_file inference/AGCN/AGCN.pdmodel \
                           --params_file inference/AGCN/AGCN.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

输出示例如下:

```
Current video file: data/fsd10/example_skeleton.npy
        top-1 class: 0
        top-1 score: 0.8932635188102722
```

可以看到，使用在FSD-10上训练好的AGCN模型对`data/example_skeleton.npy`进行预测，输出的top1类别id为`0`，置信度为0.89。

## 参考论文

- [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455), Sijie Yan, Yuanjun Xiong, Dahua Lin

- [Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1805.07694), Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu

- [Skeleton-Based Action Recognition with Multi-Stream Adaptive Graph Convolutional Networks](https://arxiv.org/abs/1912.06971), Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu

- Many thanks to [li7819559](https://github.com/li7819559) and [ZhaoJingjing713](https://github.com/ZhaoJingjing713) for contributing the code.
