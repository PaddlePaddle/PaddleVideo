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

我们对[TSM模型](./tsm.md)进行了改进，提出了**PPTSM**高精度2D实用视频分类模型。在不增加参数量和计算量的情况下，在UCF-101、Kinetics-400等数据集上精度显著超过原文。模型优化解析请参考[**pptsm实用视频模型优化解析**](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/tutorials/pp-tsm.md)。

<p align="center">
<img src="../../../images/acc_vps.jpeg" height=400 width=650 hspace='10'/> <br />
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

通过指定不同的配置文件，可以使用不同的数据格式/数据集进行训练，UCF-101数据集使用4卡训练，frames格式数据的训练启动命令如下:

```bash
python -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/tsm/pptsm.yaml
```

Kinetics400数据集使用8卡训练，frames格式数据的训练启动命令如下:

```bash
python -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/tsm/pptsm_k400.yaml
```

- 通过`-c`指定模型训练参数配置文件，默认配置文件与数据集的对应关系如下(不同数据集，数据处理部分参数配置稍有不同):

```
configs/recognition/tsm/pptsm.yaml        --> UCF-101 frames格式训练
configs/recognition/tsm/todo.yaml         --> UCF-101 videos格式训练
configs/recognition/tsm/pptsm_k400.yaml   --> Kinetics-400 frames格式训练
configs/recognition/tsm/todo.yaml         --> Kinetics-400 videos格式训练
```

- 如若进行finetune，请下载PaddleVideo的已发布模型[ppTSM.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/ppTSM/ppTSM.pdparams)，通过`--weights`指定权重存放路径。 

- 您可以自定义修改参数配置，参数用法请参考[config](../../tutorials/config.md)。


## 模型测试

```bash
python3 main.py --test -c configs/recognition/tsm/pptsm.yaml -w output/ppTSM/ppTSM_best.pdparams
```

- 通过`-c`参数指定配置文件，可下载已发布模型[ppTSM.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/ppTSM/ppTSM.pdparams)，通过`-w`指定权重存放路径进行模型测试。


当取如下参数时，在Kinetics400的验证集下评估精度如下:

| seg\_num | target\_size | Top-1 |
| :------: | :----------: | :----: |
| 8 | 224 | 0.735 |

UCF101验证集(split1)上的评估精度如下：

| seg\_num | target\_size | Top-1 |
| :------: | :----------: | :----: |
| 8 | 224 | 0.8997 |

## 模型推理

### 导出inference模型

```bash
python3 tools/export_model.py -c configs/recognition/tsm/pptsm_k400.yaml \
                              -p output/ppTSM/ppTSM_best.pdparams \
                              -o inference/ppTSM
```

上述命令将生成预测所需的模型结构文件`ppTSM.pdmodel`和模型权重文件`ppTSM.pdiparams`。

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3 tools/predict.py --video_file data/example.avi \
                         --model_file inference/ppTSM/ppTSM.pdmodel \
                         --params_file inference/ppTSM/ppTSM.pdiparams \
                         --use_gpu=True \
                         --use_tensorrt=False
```

输出示例如下:

```
Current video file: data/example.avi
	top-1 class: 5
	top-1 score: 0.9621570706367493
```

可以看到，使用在Kinetics-400上训练好的ppTSM模型对`data/example.avi`进行预测，输出的top1类别id为`5`，置信度为0.962。通过查阅类别id与名称对应表`data/k400/Kinetics-400_label_list.txt`，可知预测类别名称为`archery`。 

## 参考论文

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), Geoffrey Hinton, Oriol Vinyals, Jeff Dean
