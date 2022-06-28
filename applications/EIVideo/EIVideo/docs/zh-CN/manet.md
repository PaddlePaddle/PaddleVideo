[English](../en/manet.md) | 简体中文

# Ma-Net视频切分模型

## 内容

- [模型简介](#模型简介)

- [数据准备](#数据准备)

- [模型训练](#模型训练)

- [模型测试](#模型测试)

- [模型推理](#模型推理)

在开始使用之前，您需要按照以下命令安装额外的依赖包：
```bash
python -m pip install scikit-image
```

## 模型简介

这是CVPR2020论文"[Memory aggregation networks for efficient interactive video object segmentation](https://arxiv.org/abs/2003.13246)"的Paddle实现。

![avatar](../../../images/1836-teaser.gif)

此代码目前支持在 DAVIS 数据集上进行模型测试和模型训练，并且将在之后提供对任何给定视频的模型推理。


## 数据准备

下载 [DAVIS2017](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) 和 [scribbles](https://data.vision.ee.ethz.ch/csergi/share/DAVIS-Interactive/DAVIS-2017-scribbles-trainval.zip) 到一个文件夹中。请参阅 [DAVIS](https://davischallenge.org/davis2017/code.html).

如果您需要文件"DAVIS2017/ImageSets/2017/v_a_l_instances.txt"，请参阅链接 https://drive.google.com/file/d/1aLPaQ_5lyAi3Lk3d2fOc_xewSrfcrQlc/view?usp=sharing


## 模型训练

#### 下载并添加预先训练的模型

1. 下载  [deeplabV3+ model pretrained on COCO](https://drive.google.com/file/d/15temSaxnKmGPvNxrKPN6W2lSsyGfCtTB/view?usp=sharing) 作为主干初始化参数，或通过 wget 下载

   ```bash
   wget https://drive.google.com/file/d/15temSaxnKmGPvNxrKPN6W2lSsyGfCtTB/view?usp=sharing
   ```

2. 打开 `PaddleVideo/configs/segmentationer/manet_stage1.yaml，然后在`pretrained:`填写下载的模型权重存储路径

   ```yaml
   MODEL: #MODEL field
       framework: "Manet"
       backbone:
           name: "DeepLab"
           pretrained: fill in the path here
   ```

#### Start training

- 我们的训练过程包括两个阶段。

  - 您可以通过以下命令使用一张卡开始第一阶段的训练：

    ```bash
    python main.py -c configs/segmentation/manet.yaml
    ```

  - 使用多张卡进行训练，以加快训练过程。训练开始命令如下：

    ```bash
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    python -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_manet_stage1 main.py -c configs/segmentation/manet.yaml
    ```

  - 使用混合精度训练以加快训练过程。训练开始命令如下：

    ```bash
    export FLAGS_conv_workspace_size_limit=800 # MB
    export FLAGS_cudnn_exhaustive_search=1
    export FLAGS_cudnn_batchnorm_spatial_persistent=1

    # frames data format
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4" --log_dir=log_manet_stage1 main.py --amp -c configs/segmentation/manet.yaml
    ```

- 使用第一阶段的模型训练结果，您可以使用一张卡开始训练第二阶段（其他训练方法，如多张卡或混合精度类似于上述），命令如下：

  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3

  python -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_manet_stage1 main.py  --validate -c configs/segmentation/manet_stage2.yaml
  ```

- 此外，您可以自定义和修改参数配置，以达到在不同数据集上训练/测试的目的。建议配置文件的命名方法是 `model_dataset name_file format_data format_sampling method.yaml` ，请参考 [config](../../tutorials/config.md) 配置参数的方法。




## 模型测试

您可以通过以下命令开始测试：

```bash
python main.py --test -c configs/localization/bmn.yaml -w output/BMN/BMN_epoch_00009.pdparams -o DATASET.test_batch_size=1
```

- 您可以下载[我们的模型](https://drive.google.com/file/d/1JjYNha40rtEYKKKFtDv06myvpxagl5dW/view?usp=sharing) 解压缩它，并在配置文件中指定`METRIC.ground_truth_filename` 的路径。

- 参数 `-w` 用于指定模型路径，您可以下载 [我们的模型](https://drive.google.com/file/d/1JjYNha40rtEYKKKFtDv06myvpxagl5dW/view?usp=sharing) 并解压缩以进行测试。


测试精度在 DAVIS2017上:

| J@60  |  AUC  |
| :---: | :---: |
| 0.761 | 0.749 |



## 模型推理
