简体中文 | [English](../../../en/model_zoo/recognition/tsn.md)


# TSN

## 简介

Temporal Segment Network (TSN) 是视频分类领域经典的基于2D-CNN的解决方案。该方法主要解决视频的长时间行为判断问题，通过稀疏采样视频帧的方式代替稠密采样，既能捕获视频全局信息，也能去除冗余，降低计算量。最终将每帧特征平均融合后得到视频的整体特征，并用于分类。本代码实现的模型为基于单路RGB图像的TSN网络结构，Backbone采用ResNet-50结构。

详细内容请参考ECCV 2016年论文[Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859)

## 数据准备

PaddleVide提供了在K400和UCF101两种数据集上训练TSN的训练脚本。

K400数据下载及准备请参考[K400数据准备](../../dataset/K400.md)

UCF101数据下载及准备请参考[UCF101数据准备](../../dataset/ucf101.md)


## 模型训练

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

或用`-o` 参数在```run.sh```或命令行中进行添加
``` -o MODEL.backbone.pretrained="" ```
`-o` 参数用法请参考[conifg](../../config.md)

- 如若进行`finetune`或者[模型测试](#模型测试)等，请下载PaddleVideo的已发布模型[model]()<sup>coming soon</sup>, 通过`--weights`指定权重存放路径。 `--weights` 参数用法请参考[config](../../config.md)

K400 video格式训练

K400 frames格式训练

UCF101 video格式训练

UCF101 frames格式训练

## 实现细节

**数据处理：** 模型读取Kinetics-400数据集中的`mp4`数据，每条数据抽取`seg_num`段，每段抽取1帧图像，对每帧图像做随机增强后，缩放至`target_size`。

**训练策略：**

*  采用[Momentum](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/optimizer/momentum/Momentum_cn.html#momentum)优化算法训练，momentum值设定为0.9。
*  [l2_decay](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/regularizer/L2Decay_cn.html#l2decay)权重衰减系数为1e-4。
*  学习率在训练的总epoch数的15（1/3）和30（2/3）时分别做0.1倍的衰减。

**参数初始化**

TSN模型的卷积层和BN层参数采用Paddle默认的[KaimingNormal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/initializer/kaiming/KaimingNormal_cn.html#kaimingnormal)和[Constant](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/initializer/constant/Constant_cn.html#constant)初始化方法。而实际，在真正训练过程中，指定了pretrained参数后会以保存的权重进行参数初始化。
源代码可参考[TSN的参数初始化](https://github.com/PaddlePaddle/PaddleVideo/blob/main/paddlevideo/modeling/backbones/resnet.py#L251)。

Linear（FC）层的参数采用mean=0，std默认0.01的Normal初始化，关于Normal初始化方法可以参考[初始化](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/initializer/normal/Normal_cn.html)官方文档


## 模型测试

TSN采用CenterCrop的测试Mertics

```bash
python3 main.py --test --weights=""
```

- 指定`--weights`参数为下载已发布模型进行模型测试。


当取如下参数时，在Kinetics400的validation数据集下评估精度如下:

| seg\_num | target\_size | Top-1 |
| :------: | :----------: | :----: |
| 3 | 224 | 0.66 |
| 7 | 224 | 0.67 |

## 模型推理

首先导出模型，这里加载默认路径为```output/TSN```下的参数。并将预测模型导出至`inference`下。

```bash
    python3.7 tools/export_model.py -c configs/recognition/tsn/tsn.yaml -p output/TSN/TSN_best.pdparams -o ./inference
```

之后，进行模型推理

```bash
python3 tools/predict.py -v data/example.avi --model_file "./inference/TSN.pdmodel" --params_file "./inference/TSN.pdiparams" --enable_benchmark=False --model="TSN"
```

更多关于预测部署功能介绍请参考[../../tutorials/deployment.md]

## 参考论文

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool
