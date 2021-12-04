 简体中文

# MoViNet视频分类模型

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [实现细节](#实现细节)
- [参考论文](#参考论文)
- [参考实现](#参考实现)

## 模型简介

MoViNet是Google Research研发的移动视频网络。它是一个可以用于在线推理视频流的，读取视频高效而且计算迅速的视频网络的一种。
这篇模型的产出还有EfficientNet一作的参与。
主要使用以下三步渐进式的方法来实现模型的搭建：  
- 1 使用神经结构搜索的方法来搜索MoViNet空间结构
- 2 使用因果卷积算子和流缓冲区来弥补准确率的损失
- 3 Temporal Ensembles提升准确率
详细内容请参考CVPR 2021年论文 [MoViNets: Mobile Video Networks for Efficient Video Recognition](https://arxiv.org/abs/2103.11511)

## 数据准备

UCF101数据下载及准备请参考[ucf101数据准备](../../dataset/ucf101.md)

## 模型训练

### UCF-101数据集训练

#### 配置模型名称

配置 `PaddleVideo/configs/recognition/movinet/movineta0_ucf101_frame.yaml`

   ```yaml
    MODEL: #MODEL field
      framework: "MoViNetRecognizerFrame" #读取逐帧输出的图片文件
      backbone: #Mandatory, indicate the type of backbone, associate to the 'paddlevideo/modeling/backbones/' .
        name: "MoViNet" #Mandatory, The name of backbone.
        causal: True # 是否使用因果卷积
        pretrained: False
        num_classes: 101 # 类别数，UCF-101一致
        cfg: "A0"
      head:
        name: "MoViNetHead" #Mandatory, indicate the type of head, associate to the 'paddlevideo/modeling/heads'
        num_classes: 101
        in_channels: 2048
    DATASET: #DATASET field
      batch_size: 32 #Mandatory, bacth size
      num_workers: 0 #Mandatory, XXX the number of subprocess on each GPU.
      train:
        format: "FrameDataset" #Mandatory, indicate the type of dataset, associate to the 'paddlevidel/loader/dateset'
        data_prefix: "" #Mandatory, train data root path
        file_path: "ucf101_train_split_1_rawframes.txt" # 修改为之前数据处理代码生成的训练集frame文件路径
        suffix: 'img_{:05}.jpg'
      valid:
        format: "FrameDataset" #Mandatory, indicate the type of dataset, associate to the 'paddlevidel/loader/dateset'
        data_prefix: "" #Mandatory, valid data root path
        file_path: "ucf101_val_split_1_rawframes.txt" # 修改为之前数据处理代码生成的验证集frame文件路径
        suffix: 'img_{:05}.jpg'
      test:
        format: "FrameDataset" #Mandatory, indicate the type of dataset, associate to the 'paddlevidel/loader/dateset'
        data_prefix: "" #Mandatory, valid data root path
        file_path: "ucf101_val_split_1_rawframes.txt" # 修改为之前数据处理代码生成的验证集frame文件路径
        suffix: 'img_{:05}.jpg'
   ```

#### 开始训练

!python main.py -c configs/recognition/movinet/movinet_ucf101_frame.yaml --validate


## 模型测试


- 若需单独运行测试代码，其启动命令如下：

```bash
python3.7 main.py --test -c configs/recognition/movinet/movinet_ucf101_frame.yaml -w output/MoViNet/MoViNet_best.pdparams
```
- 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。

---

当测试配置采用如下参数时，在UCF-101的validation数据集上的评估精度如下：

| backbone | Sampling method | num_seg | target_size | Top-1 | checkpoints |
| :------: | :-------------: | :-----------------: | :-----: | :---------: | :---: |
| A0 |     Uniform     |    8    |     224     |  |      |
| A1 |     Uniform     |    8    |     224     |  |      |
| A2 |     Uniform     |    8    |     224     | |        |
| A3 |     Uniform     |    8    |     224     |  |      |
| A4 |     Uniform     |    8    |     224     |  |      |
| A5 |     Uniform     |    8    |     224     |  |        |

## 实现细节

**数据处理**

- 模型读取UCF-101数据集划分好的frame文件路径

**训练策略**

- 采用Momentum优化算法训练，momentum=0.9
- 采用L2_Decay，权重衰减系数为1e-4
- 采用全局梯度裁剪，裁剪系数为20.0


**参数初始化**

- KaimingNormal对3D卷积核进行初始化

## 参考论文

- [MoViNets: Mobile Video Networks for Efficient Video Recognition](https://arxiv.org/abs/2103.11511)

## 参考实现

- [MoViNetPytorch](https://github.com/Atze00/MoViNet-pytorch)
