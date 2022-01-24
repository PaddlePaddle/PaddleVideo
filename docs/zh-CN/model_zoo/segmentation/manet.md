[English](../../../en/model_zoo/segmentation/manet.md) | 简体中文

# Ma-Net视频切分模型

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)




## 模型简介

这是CVPR2020论文"[Memory aggregation networks for efficient interactive video object segmentation](https://arxiv.org/abs/2003.13246)"的Paddle实现。

![avatar](../../../images/1836-teaser.gif)

此代码目前支持在 DAVIS 数据集上进行模型测试和模型训练，并且将在之后提供对任何给定视频的模型推理。


## 数据准备

DAVIS数据下载及准备请参考[DAVIS2017数据准备](../../dataset/DAVIS2017.md)


## 模型训练

### 下载并添加预先训练的模型

1. 下载  [deeplabV3+ model pretrained on COCO](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/DeeplabV3_coco.pdparams) 作为主干初始化参数，或通过 wget 下载

   ```bash
   wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/DeeplabV3_coco.pdparams
   ```

2. 打开 `PaddleVideo/configs/segmentationer/manet_stage1.yaml，然后在`pretrained:`填写下载的模型权重存储路径

   ```yaml
   MODEL: #MODEL field
       framework: "ManetSegment_Stage1"
       backbone:
           name: "DeepLab"
           pretrained: fill in the path here
   ```

### 开始训练

- 我们的训练过程包括两个阶段。

  - 您可以通过以下命令使用一张卡开始第一阶段的训练：

    ```bash
    python main.py -c configs/segmentation/manet_stage1.yaml
    ```

- 使用第一阶段的模型训练结果，您可以使用一张卡开始训练第二阶段（其他训练方法，如多张卡或混合精度类似于上述），命令如下：

  ```bash
    python main.py -c configs/segmentation/manet_stage2.yaml
  ```

- 此外，您可以自定义和修改参数配置，以达到在不同数据集上训练/测试的目的。建议配置文件的命名方法是 `model_dataset name_file format_data format_sampling method.yaml` ，请参考 [config](../../tutorials/config.md) 配置参数的方法。




## 模型测试

您可以通过以下命令开始测试：

  ```bash
    python main.py --test -c configs/segmentation/manet_stage2.yaml -w output/ManetSegment_Stage2/ManetSegment_Stage2_step_100001.pdparams  
  ```

- 您可以下载[我们的模型](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/MANet_davis2017.pdparams) 解压缩它以用于测试，使用参数 `-w` 用于指定模型路径。

- 同时在配置文件中指定`METRIC.ground_truth_filename` 的路径。


测试精度在 DAVIS2017上:

| J@60  |  AUC  |
| :---: | :---: |
| 0.761 | 0.749 |
