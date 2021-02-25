[English](../../../en/model_zoo/recognition/tsm.md) | 简体中文

# TSM视频分类模型

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)

## 模型介绍

Temporal Shift Module (TSM) 是当前比较受关注的模型，通过通道移动的方法在不增加任何额外参数量和计算量的情况下极大的提升了模型对于视频时间信息的利用能力，并且由于其具有轻量高效的特点，十分适合工业落地。

<div align="center">
<img src="../../../images/tsm_architecture.png" height=250 width=700 hspace='10'/> <br />
</div>

本代码实现的模型为基于单路RGB图像的TSM网络结构，Backbone采用ResNet-50结构。

详细内容请参考ICCV 2019年论文 [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf)

## 数据准备

Kinetics400数据下载及准备请参考[k400数据准备](../../dataset/K400.md)

UCF101数据下载及准备请参考[ucf101数据准备](../../dataset/ucf101.md)


## 模型训练

### 预训练模型下载

- 加载在ImageNet1000上训练好的ResNet50权重作为Backbone初始化参数，请下载此[模型参数](https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams),
或是通过命令行下载

```bash
wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams
```

并将路径添加到configs中backbone字段下

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
python -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_tsm  main.py  --validate -c configs/recognition/tsm/tsm.yaml
```

- 如若进行finetune，请下载PaddleVideo的已发布模型[TSM.pdparams]()<sup>coming soon</sup>，通过`--weights`指定权重存放路径。

- 您可以自定义修改参数配置，参数用法请参考[config](../../tutorials/config.md)。


### 实现细节

**数据处理：** 模型读取Kinetics-400数据集中的`mp4`数据，每条数据抽取`seg_num`段，每段抽取1帧图像，对每帧图像做随机增强后，缩放至`target_size`。

**训练策略：**

*  采用Momentum优化算法训练，momentum=0.9
*  l2_decay权重衰减系数为1e-4
*  学习率在训练的总epoch数的1/3和2/3时分别做0.1倍的衰减

**参数初始化**<sup>coming soon</sup>

## 模型测试

```bash
python3 main.py --test -c configs/recognition/tsm/tsm.yaml -w output/TSM/TSM_best.pdparams
```

- 指定`--weights`参数，下载已发布模型[TSM.pdparams]()<sup>coming soon</sup>进行模型测试


当取如下参数时，在Kinetics400的validation数据集下评估精度如下:

| seg\_num | target\_size | Top-1 |
| :------: | :----------: | :----: |
| 8 | 224 | 0.70 |

## 模型推理

### 导出inference模型

```bash
python3 tools/export_model.py -c configs/recognition/tsm/tsm.yaml \
                              -p output/TSM/TSM_best.pdparams \
                              -o inference/TSM
```

上述命令将生成预测所需的模型结构文件`TSM.pdmodel`和模型权重文件`TSM.pdiparams`。

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3 tools/predict.py --video_file data/example.avi \
                         --model_file inference/TSM/TSM.pdmodel \
                         --params_file inference/TSM/TSM.pdiparams \
                         --use_gpu=True \
                         --use_tensorrt=False
```

## 参考论文

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
