[English](../../../en/model_zoo/multimodal/actbert.md) | 简体中文

# ActBERT多模态预训练模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [参考论文](#参考论文)

在开始使用之前，您需要按照以下命令安装额外的依赖包：
```bash
python -m pip install paddlenlp
python -m pip install lmdb
```

## 模型简介

ActBERT是百度在CVPR2020提出的多模态预训练模型，它结合输入文本、图像和视频动作三种模态，使用一种全新的纠缠编码模块从三个来源进行多模态特征学习，以增强两个视觉输入和语言之间的互动功能。模型采用RandomMask和NSP的方式进行训练，在文本视频搜索、视频描述生成等5个下游任务中表现优异。

<div align="center">
<img src="../../../images/actbert.png" height=400 width=500 hspace='10'/> <br />
</div>


## 数据准备

HowTo100M数据下载及准备请参考[HowTo100M数据准备](../../dataset/howto100m.md)

MSR-VTT数据下载及准备请参考[MSR-VTT数据准备](../../dataset/msrvtt.md)


## 模型训练

### HowTo100M数据集训练

#### 下载并添加预训练模型

下载BERT预训练模型[bert-base-uncased](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/bert-base-uncased.pdparams)作为Backbone初始化参数，或是通过命令行下载

```bash
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/bert-base-uncased.pdparams
```

并将文件路径添加到配置文件中的`MODEL.framework.backbone.pretrained`字段，如下：

```yaml
MODEL:
    framework: "ActBert"
    backbone:
        name: "BertForMultiModalPreTraining"
        pretrained: 将路径填写到此处
```

- 由于训练数据集过大，本代码提供小数据训练功能，训练配置仅供参考~

#### 开始训练

- 训练启动命令如下:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_actbert  main.py  --validate -c configs/multimodal/actbert/actbert.yaml
```

- 开启amp混合精度训练，可加速训练过程，其训练启动命令如下：

```bash
export FLAGS_conv_workspace_size_limit=800 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_actbert  main.py  --amp --validate -c configs/multimodal/actbert/actbert.yaml
```

- 另外您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的。


## 模型测试

- 对下游任务：文本-视频检索，在MSR-VTT数据集上评估性能，评估脚本启动方式如下：


```bash
python3.7 main.py --test -c configs/multimodal/actbert/actbert_msrvtt.yaml -w Actbert.pdparams
```

- 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。


MSR-VTT数据集测试精度:

| R@1 | R@5 | R@10 | Median R | Mean R | checkpoints |
| :------: | :----------: | :----: | :----: | :----: | :----: |
| 8.6 | 31.2 | 45.5 | 13.0 | 28.5 | [ActBERT.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ActBERT.pdparams) |


## 参考论文

- [ActBERT: Learning Global-Local Video-Text Representations
](https://arxiv.org/abs/2011.07231), Linchao Zhu, Yi Yang
