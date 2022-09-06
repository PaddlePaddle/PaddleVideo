[English](../../../en/model_zoo/localization/yowo.md) | 简体中文

# YOWO 视频动作检测模型

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介

YOWO是具有两个分支的单阶段网络。一个分支通过2D-CNN提取关键帧（即当前帧）的空间特征，而另一个分支则通过3D-CNN获取由先前帧组成的剪辑的时空特征。为准确汇总这些特征，YOWO使用了一种通道融合和关注机制，最大程度地利用了通道间的依赖性。最后将融合后的特征进行帧级检测。

<div align="center">
<img src="../../../images/yowo.jpg">
</div>


## 数据准备

UCF101-24数据下载及准备请参考[UCF101-24数据准备](../../dataset/ucf24.md)


## 模型训练

### UCF101-24数据集训练

#### 下载并添加预训练模型

1. 下载预训练模型 [resnext101_kinetics](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/resnext101_kinetics.pdparams) 和 [darknet](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/darknet.pdparam) 作为Backbone初始化参数，或通过wget命令下载

   ```bash
    wget -nc https://videotag.bj.bcebos.com/PaddleVideo-release2.3/darknet.pdparam
    wget -nc https://videotag.bj.bcebos.com/PaddleVideo-release2.3/resnext101_kinetics.pdparams
   ```

2. 打开`PaddleVideo/configs/localization/yowo.yaml`，将下载好的权重存放路径分别填写到下方`pretrained_2d:`和`pretrained_3d:`之后

    ```yaml
    MODEL:
        framework: "YOWOLocalizer"
        backbone:
            name: "YOWO"
            num_class: 24
            pretrained_2d: 将2D预训练模型路径填写到此处
            pretrained_3d: 将3D预训练模型路径填写到此处
    ```

#### 开始训练

- UCF101-24数据集使用单卡训练，训练方式的启动命令如下:

    ```bash
    python3 main.py -c configs/localization/yowo.yaml --validate --seed=1
    ```
    
- 开启amp混合精度训练，可加速训练过程，其训练启动命令如下：

    ```bash
    python3 main.py --amp -c configs/localization/yowo.yaml --validate --seed=1
    ```
    
- 另外您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的，建议配置文件的命名方式为`模型_数据集名称_文件格式_数据格式_采样方式.yaml`，参数用法请参考[config](../../contribute/config.md)。


## 模型测试

- YOWO 模型在训练时同步进行验证，您可以通过在训练日志中查找关键字`best`获取模型测试精度，日志示例如下:

  ```
  Already save the best model (fsocre)0.8779
  ```

- 由于 YOWO 模型测试模式的评价指标为的**Frame-mAP (@ IoU 0.5)**，与训练过程中验证模式采用的**fscore**不同，所以训练日志中记录的验证指标`fscore`不代表最终的测试分数，因此在训练完成之后可以用测试模式对最好的模型进行测试获取最终的指标，命令如下：

  ```bash
  python3 main.py -c configs/localization/yowo.yaml --test --seed=1 -w 'output/YOWO/YOWO_epoch_00005.pdparams'
  ```


  当测试配置采用如下参数时，在UCF101-24的test数据集上的测试指标如下：

  | Model    | 3D-CNN backbone | 2D-CNN backbone | Dataset  |Input    | Frame-mAP <br>(@ IoU 0.5)    |   checkpoints  |
  | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
  | YOWO | 3D-ResNext-101 | Darknet-19 | UCF101-24 | 16-frames, d=1 | 80.83 | [YOWO.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/YOWO_epoch_00005.pdparams) |


## 模型推理

### 导出inference模型

```bash
python3 tools/export_model.py -c configs/localization/yowo.yaml -p 'output/YOWO/YOWO_epoch_00005.pdparams'
```

上述命令将生成预测所需的模型结构文件`YOWO.pdmodel`和模型权重文件`YOWO.pdiparams`。

- 各参数含义可参考[模型推理方法](../../usage.md#5-模型推理)

### 使用预测引擎推理

```bash
python3 tools/predict.py -c configs/localization/yowo.yaml -i 'data/ucf24/HorseRiding.avi' --model_file ./inference/YOWO.pdmodel --params_file ./inference/YOWO.pdiparams
```

输出示例如下（可视化）:

<div align="center">
  <img  src="../../../images/horse_riding.gif" alt="Horse Riding">
</div>

可以看到，使用在UCF101-24上训练好的YOWO模型对```data/ucf24/HorseRiding.avi```进行预测，每张帧输出的类别均为HorseRiding，置信度为0.80左右。

## 参考论文

- [You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization](https://arxiv.org/pdf/1911.06644.pdf), Köpüklü O, Wei X, Rigoll G.