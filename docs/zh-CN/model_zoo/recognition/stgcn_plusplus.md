[English](../../../en/model_zoo/recognition/stgcn_plusplus.md) | 简体中文

# STGCN++基于骨骼的行为识别模型

---
## 内容

- [STGCN++基于骨骼的行为识别模型](#stgcn基于骨骼的行为识别模型)
  - [内容](#内容)
  - [模型简介](#模型简介)
  - [数据准备](#数据准备)
  - [模型训练](#模型训练)
    - [NTURGB+D数据集训练](#nturgbd数据集训练)
  - [模型测试](#模型测试)
    - [NTURGB+D数据集模型测试](#nturgbd数据集模型测试)
  - [模型推理](#模型推理)
    - [导出inference模型](#导出inference模型)
    - [使用预测引擎推理](#使用预测引擎推理)
  - [参考论文](#参考论文)


## 模型简介


人体骨架作为人类行为的一种简洁的表现形式，近年来受到越来越多的关注。许多基于骨架的动作识别方法都采用了图卷积网络（GCN）来提取人体骨架上的特征。尽管在以前的工作中取得了积极的成果，但基于GCN的方法在健壮性、互操作性和可扩展性方面受到限制。本文作者提出了一个原始的GCN模型ST-GCN++。仅对原始ST-GCN进行简单修改。ST-GCN++重新设计了空间模块和时间模块，ST-GCN++就获得了与具有复杂注意机制的SOTA识别性能。同时，计算开销也减大大的减少了。

## 数据准备

NTURGB+D数据集来自mmaction项目,地址如下:

[https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl)

或者AI Studio下载地址:

[https://aistudio.baidu.com/aistudio/datasetdetail/167195](https://aistudio.baidu.com/aistudio/datasetdetail/167195)

## 模型训练

### NTURGB+D数据集训练

- 数据集使用单卡训练，启动命令如下:

```bash
cd PaddleVideo
pip install -r requirements.txt
ln -s path/to/ntu60_hrnet.pkl data/ntu60_hrnet.pkl
python -u main.py --validate \
-c configs/recognition/stgcn_plusplus/stgcn_plusplus_ntucs.yaml
```



- 您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的，参数用法请参考[config](../../tutorials/config.md)。


## 模型测试

### NTURGB+D数据集模型测试

- 模型测试的启动命令如下：

```bash
python -u main.py --test -c configs/recognition/stgcn_plusplus/stgcn_plusplus_ntucs.yaml \
                  --weights STGCN_PlusPlus_best.pdparams
```

- 通过`-c`参数指定配置文件，通过`--weights`指定权重存放路径进行模型测试。


模型在NTURGB+D数据集上baseline实验精度如下:

| Test_Data | Top-1 | checkpoints |
| :----: |:-----:| :---- |
| NTURGB+D | 97.56 | [STGCN_PlusPlus_best.pdparams](https://aistudio.baidu.com/aistudio/datasetdetail/169754) |



## 模型推理

### 导出inference模型

```bash
python tools/export_model.py -c configs/recognition/stgcn_plusplus/stgcn_plusplus_ntucs.yaml \
                                 --save_name inference \
                                 -p=STGCN_PlusPlus_best.pdparams \
                                 -o=./output/STGCN_PlusPlus/
```

上述命令将生成预测所需的模型结构文件`inference.pdmodel`和模型权重文件`inference.pdiparams`。

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python tools/predict.py --config configs/recognition/stgcn_plusplus/stgcn_plusplus_ntucs.yaml \
                        --use_gpu=True \
                        --model_file=./output/STGCN_PlusPlus/inference.pdmodel \
                        --params_file=./output/STGCN_PlusPlus/inference.pdiparams --batch_size=1 \
                        --input_file=./data/stdgcn_plusplus_data/example_ntu60_skeleton.pkl
```

输出示例如下:

```
Current video file: ./data/stdgcn_plusplus_data/example_ntu60_skeleton.pkl
	top-1 class: 0
	top-1 score: 0.9153057932853699
```

可以看到，使用在NTURGB+D数据集上训练好的STGCN++模型对`data/stdgcn_plusplus_data/example_ntu60_skeleton.pkl`进行预测，输出的top1类别id为`0`，置信度为0.915。

## 参考论文

- [PYSKL: Towards Good Practices for Skeleton Action Recognition](https://arxiv.org/pdf/2205.09443.pdf), Haodong Duan, Jiaqi Wang, Kai Chen, Dahua Lin
