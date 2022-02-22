[English](../../../en/model_zoo/segmentation/asrf.md) | 简体中文

# ASRF 视频动作分割模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)

## 模型简介

ASRF模型是在视频动作分割模型MS-TCN上的改进，发表在2021年的WACV上。我们对官方实现的pytorch代码进行复现，在PaddleVideo获得了近似的结果。

<p align="center">
<img src="../../../images/asrf.png" height=300 width=400 hspace='10'/> <br />
ASRF Overview
</p>

## 数据准备

ASRF的训练数据可以选择50salads、breakfast、gtea三个数据集，数据下载及准备请参考[视频动作分割数据集](../../dataset/SegmentationDataset.md)

不同于MS-TCN，ASRF模型需要额外的数据构建，脚本流程如下
```bash
python data/50salads/prepare_asrf_data.py --dataset_dir data/
```

## 模型训练

数据准备完毕后，可以通过如下方式启动训练：

```bash
# gtea数据集
export CUDA_VISIBLE_DEVICES=3
python3.7 main.py  --validate -c configs/segmentation/asrf/asrf_gtea.yaml --seed 1538574472
```

- 从头开始训练，使用上述启动命令行或者脚本程序即可启动训练，不需要用到预训练模型，视频动作分割模型通常为全卷积网络，由于视频的长度不一，故视频动作分割模型的batch_size字段通常设为1，即不需要批量训练，目前也仅支持**单样本**训练

## 模型测试

可通过如下方式进行模型测试：

```bash
python main.py  --test -c configs/segmentation/asrf/asrf_gtea.yaml --weights=./output/ASRF/ASRF_split_1.pdparams
```

- 指标的具体实现是参考MS-TCN作者[evel.py](https://github.com/yabufarha/ms-tcn/blob/master/eval.py)提供的测试脚本，计算Acc、Edit和F1分数。

- pytorch的复现来源于官方提供的[代码库](https://github.com/yiskw713/asrf)

- 数据集的评估方法采用MS-TCN论文中的折交验证方法，而折交的划分方式与MS-TCN论文中相同。

在Breakfast数据集下评估精度如下(采用4折交验证):

| Model | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| paper | 67.6% | 72.4% | 74.3% | 68.9% | 56.1% |
| pytorch | 65.8% | 71.0% | 72.3% | 66.5% | 54.9% |
| paddle | 66.1% | 71.9% | 73.3% | 67.9% | 55.7% |

在50salads数据集下评估精度如下(采用5折交验证):

| Model | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| paper | 84.5% | 79.3% | 82.9% | 83.5% | 77.3% |
| pytorch | 81.4% | 75.6% | 82.7% | 81.2% | 77.2% |
| paddle | 81.6% | 75.8% | 83.0% | 81.5% | 74.8% |

在gtea数据集下评估精度如下(采用4折交验证):

| Model | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| paper | 77.3% | 83.7% | 89.4% | 87.8% | 79.8% |
| pytorch | 76.3% | 79.6% | 87.3% | 85.8% | 74.9% |
| paddle | 77.1% | 83.3% | 88.9% | 87.5% | 79.1% |

给出在gtea数据集下的折交的模型权重

Test_Data| F1@0.5 | checkpoints |
| :----: | :----: | :---- |
| gtea_split1 | 72.4409 | [ASRF_gtea_split_1.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ASRF_gtea_split_1.pdparams) |
| gtea_split2 | 76.6666 | [ASRF_gtea_split_2.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ASRF_gtea_split_2.pdparams) |
| gtea_split3 | 84.5528 | [ASRF_gtea_split_3.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ASRF_gtea_split_3.pdparams) |
| gtea_split4 | 82.6771 | [ASRF_gtea_split_4.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ASRF_gtea_split_4.pdparams) |


## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/segmentation/asrf/asrf_gtea.yaml \
                                -p data/ASRF_gtea_split_1.pdparams \
                                -o inference/ASRF
```

上述命令将生成预测所需的模型结构文件`ASRF.pdmodel`和模型权重文件`ASRF.pdiparams`。

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

输入预测模型的txt文件为需要预测的文件列表，如:
```
S1_Cheese_C1.npy
S1_CofHoney_C1.npy
S1_Coffee_C1.npy
S1_Hotdog_C1.npy
...
```

```bash
python3.7 tools/predict.py --input_file data/gtea/splits/test.split1.bundle \
                           --config configs/segmentation/asrf/asrf_gtea.yaml \
                           --model_file inference/ASRF/ASRF.pdmodel \
                           --params_file inference/ASRF/ASRF.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

输出示例如下:

```bash
result write in : ./inference/infer_results/S1_Cheese_C1.txt
result write in : ./inference/infer_results/S1_CofHoney_C1.txt
result write in : ./inference/infer_results/S1_Coffee_C1.txt
result write in : ./inference/infer_results/S1_Hotdog_C1.txt
result write in : ./inference/infer_results/S1_Pealate_C1.txt
result write in : ./inference/infer_results/S1_Peanut_C1.txt
result write in : ./inference/infer_results/S1_Tea_C1.txt
```


## 参考论文

- [Alleviating Over-segmentation Errors by Detecting Action Boundaries](https://arxiv.org/pdf/2007.06866v1.pdf), Yuchi Ishikawa, Seito Kasai, Yoshimitsu Aoki, Hirokatsu Kataoka
