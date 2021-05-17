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

Temporal Shift Module (TSM) 是当前比较受关注的视频分类模型，通过通道移动的方法在不增加任何额外参数量和计算量的情况下，极大地提升了模型对于视频时间信息的利用能力，并且由于其具有轻量高效的特点，十分适合工业落地。

<div align="center">
<img src="../../../images/tsm_architecture.png" height=250 width=700 hspace='10'/> <br />
</div>



本代码实现的模型为**基于单路RGB图像**的TSM网络，Backbone采用ResNet-50结构。

详细内容请参考ICCV 2019年论文 [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf)

## 数据准备

Kinetics400数据下载及准备请参考[k400数据准备](../../dataset/K400.md)

UCF101数据下载及准备请参考[ucf101数据准备](../../dataset/ucf101.md)


## 模型训练

### 预训练模型下载

1. 加载在ImageNet1000上训练好的ResNet50权重作为Backbone初始化参数[ResNet50_pretrain.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams)，
   也可以通过命令行下载
   
   ```bash
   wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams
   ```

2. 将下载好的权重路径填写到下方`pretrained:`之后

   ```bash
   MODEL:
       framework: "Recognizer2D"
       backbone:
           name: "ResNetTSM"
           pretrained: 将路径填写到此处
   ```

### 开始训练

- 通过指定不同的配置文件，可以使用不同的数据格式/数据集进行训练，以Kinetics-400数据集 + 8卡 + frames格式的训练配置为例，启动命令如下（更多的训练命令在`PaddleVideo/run.sh`中可以查看）。

  ```bash
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7,8" --log_dir=log_tsm main.py  --validate -c configs/recognition/tsm/tsm_k400_frames.yaml
  ```

- 如若进行finetune（以UCF-101 + 4卡 + frames格式的训练配置为例），请先下载PaddleVideo的已发布模型权重[TSM.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/TSM/TSM.pdparams)，再将`PaddleVideo/configs/recognition/tsm/tsm_ucf101_frames.yaml`中`pretrained:`后的字段替换成下载好的`TSM.pdparams`路径，最后执行下方的命令，启动finetune训练。

  ```bash
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_tsm main.py  --validate -c configs/recognition/tsm/tsm_ucf101_frames.yaml
  ```

- 如若要继续训练，则指定`--weights`参数为要继续训练的模型即可

  ```bash
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_tsm main.py  --validate -c configs/recognition/tsm/tsm_ucf101_frames.yaml --weights resume_model.pdparams
  ```

- 另外您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的，具体参数用法请参考[config](../../tutorials/config.md)。


### 实现细节

**数据处理**

- 模型读取Kinetics-400数据集中的`mp4`数据，首先将每条视频数据划分成`seg_num`段，然后均匀地从每段中抽取1帧图像，得到稀疏采样的`seg_num`张视频帧，再对这`seg_num`帧图像做同样的随机数据增强，包括多尺度的随机裁剪、随机左右翻转、数据归一化等，最后缩放至`target_size`。

**训练策略**

*  采用Momentum优化算法训练，momentum=0.9
*  采用L2_Decay，权重衰减系数为1e-4
*  采用全局梯度裁剪，裁剪系数为20.0
*  总epoch数为50，学习率在epoch达到20、40进行0.1倍的衰减
*  FC层的权重与偏置的学习率分别为为整体学习率的5倍、10倍，且偏置不设置L2_Decay
*  Dropout_ratio=0.5

**参数初始化**

- 以$N(0,0.001)$的正态分布来初始化FC层的权重，以常数0来初始化FC层的偏置

## 模型测试

将待测的模型权重放到`output/TSM/`目录下，测试命令如下

```bas
python3.7 main.py --test -c configs/recognition/tsm/tsm_k400_frames.yaml --weights output/TSM/TSM_best.pdparams
```

- 也可以使用我们下载我们训练好并已发布的模型[TSM.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/TSM/TSM.pdparams)进行测试


当测试配置采用如下参数时，在Kinetics-400的validation数据集上的评估精度如下:

| seg\_num | target\_size | Top-1  |
| :------: | :----------: | :----: |
|    8     |     224      | 0.7106 |

## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/recognition/tsm/tsm_k400_frames.yaml \
                              -p output/TSM/TSM_best.pdparams \
                              -o inference/TSM
```

上述命令将生成预测所需的模型结构文件`TSM.pdmodel`和模型权重文件`TSM.pdiparams`。

各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --video_file data/example.avi \
                         --model_file inference/TSM/TSM.pdmodel \
                         --params_file inference/TSM/TSM.pdiparams \
                         --use_gpu=True \
                         --use_tensorrt=False
```

## 参考论文

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han

