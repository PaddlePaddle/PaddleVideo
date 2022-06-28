[English](../../../en/model_zoo/localization/bmn.md) | 简体中文

# BMN 视频动作定位模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介

BMN模型是百度自研，2019年ActivityNet夺冠方案，为视频动作定位问题中proposal的生成提供高效的解决方案，在PaddlePaddle上首次开源。此模型引入边界匹配(Boundary-Matching, BM)机制来评估proposal的置信度，按照proposal开始边界的位置及其长度将所有可能存在的proposal组合成一个二维的BM置信度图，图中每个点的数值代表其所对应的proposal的置信度分数。网络由三个模块组成，基础模块作为主干网络处理输入的特征序列，TEM模块预测每一个时序位置属于动作开始、动作结束的概率，PEM模块生成BM置信度图。

AI Studio项目使用链接：[ActivityNet Challenge 2019 冠军模型：BMN](https://aistudio.baidu.com/aistudio/projectdetail/2250674?contributionType=1)

<p align="center">
<img src="../../../images/BMN.png" height=300 width=400 hspace='10'/> <br />
BMN Overview
</p>

## 数据准备

BMN的训练数据采用ActivityNet1.3提供的数据集，数据下载及准备请参考[ActivityNet数据说明](../../dataset/ActivityNet.md)

## 模型训练

数据准备完毕后，可以通过如下方式启动训练：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_bmn main.py  --validate -c configs/localization/bmn.yaml
```

- 从头开始训练，使用上述启动命令行或者脚本程序即可启动训练，不需要用到预训练模型

### 单卡训练

单卡训练请将配置文件中的`DATASET.batch_size`字段修改为16，如下:

```yaml
DATASET: #DATASET field
  batch_size: 16 #single card bacth size
```

单卡训练启动方式如下:

```bash
python -B main.py  --validate -c configs/localization/bmn.yaml
```


## 模型测试

可通过如下方式进行模型测试:

```bash
python main.py --test -c configs/localization/bmn.yaml -w output/BMN/BMN_epoch_00009.pdparams -o DATASET.test_batch_size=1
```

- 目前仅支持**单卡**， `batch_size`为**1**进行模型测试，

-  请下载[activity\_net\_1\_3\_new.json](https://paddlemodels.bj.bcebos.com/video_detection/activity_net_1_3_new.json)文件，并通过`METRIC.ground_truth_filename`字段指定该ground_truth文件，相较于原始的activity\_net.v1-3.min.json文件，我们过滤了其中一些失效的视频条目。

- 通过 `-w`参数指定待测试模型文件的路径，您可以下载我们训练好的模型进行测试[BMN.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/BMN/BMN.pdparams)

- 上述程序会将运行结果保存在配置文件`METRIC.output_path`字段指定的路径，默认为`data/bmn/BMN_Test_output`文件夹下，测试结果保存在配置文件`METRIC.result_path`字段指定的文件，默认为`data/bmn/BMN_Test_results/bmn_results_validation.json`文件。

- 我们基于ActivityNet官方提供的测试脚本，计算AR@AN和AUC。具体计算过程请参考[anet_prop.py](https://github.com/PaddlePaddle/PaddleVideo/blob/main/paddlevideo/metrics/ActivityNet/anet_prop.py)文件。

- 注：评估时可能会出现loss为nan的情况。这是由于评估时用的是单个样本，可能存在没有iou>0.6的样本，所以为nan，对最终的评估结果没有影响。

在ActivityNet1.3数据集下评估精度如下:

| AR@1 | AR@5 | AR@10 | AR@100 | AUC |
| :---: | :---: | :---: | :---: | :---: |
| 33.26 | 49.48 | 56.86 | 75.19 | 67.23% |


## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/localization/bmn.yaml \
                                -p data/BMN.pdparams \
                                -o inference/BMN
```

上述命令将生成预测所需的模型结构文件`BMN.pdmodel`和模型权重文件`BMN.pdiparams`。

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example_feat.list \
                           --config configs/localization/bmn.yaml \
                           --model_file inference/BMN/BMN.pdmodel \
                           --params_file inference/BMN/BMN.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

- `input_file`为文本文件，指定待推断的文件信息，包括特征文件路径`feat_path`和视频时长(单位:s)`duration_second`。

输出示例如下:

```
BMN Inference results of data/example_feat.npy :
{'score': 0.7968077063560486, 'segment': [0.0, 122.9877]}
{'score': 0.49097609519958496, 'segment': [12.423000000000002, 124.23]}
{'score': 0.21395835280418396, 'segment': [39.7536, 122.9877]}
{'score': 0.2106524258852005, 'segment': [0.0, 109.3224]}
{'score': 0.06876271963119507, 'segment': [23.6037, 114.2916]}
```

- 默认只打印前5个得分最高的proposal，所有的预测结果可在输出文件中查看，默认输出文件路径为`data/bmn/BMN_INFERENCE_results`。输出路径可在配置文件中的`INFERENCE.result_path`自行修改。

## 参考论文

- [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702), Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, Shilei Wen.
