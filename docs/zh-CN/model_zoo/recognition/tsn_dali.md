[English](../../../en/model_zoo/recognition/tsn_dali.md) | 简体中文

# TSN模型-DALI训练加速

- [方案简介](#方案简介)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考文献](#参考文献)

## 方案简介
训练速度慢是视频模型训练常见的问题，PaddleVideo使用飞桨2.0的dataloader接口进行数据读取，凭借其优异的多进程加速能力，模型的训练速度可以显著增加。TSN是视频领域常用的2D模型，我们对其训练速度进行了进一步优化。基于[nvidia DALI](https://github.com/NVIDIA/DALI)的GPU解码能力，我们对nvidia DALI进行了二次开发，实现了其均匀分段的帧采样方式，进一步提升了TSN模型的训练速度。

### 性能

测试环境: 
```
机器: Tesla v100
显存: 4卡16G
Cuda: 9.0
单卡batch_size: 32
```

训练速度对比如下:

| 加速方式  | batch耗时/s  | reader耗时/s | ips:instance/sec | 加速比 | 
| :--------------- | :--------: | :------------: | :------------: | :------------: |
| DALI | 2.083 | 1.804 | 15.36597  | 1.41x | 
| Dataloader:  单卡num_workers=4 | 2.943 | 2.649 | 10.87460| base |
| pytorch实现 | TODO | TODO | TODO | TODO |


## 环境配置

我们提供docker运行环境方便您使用，基础镜像为:

```
    huangjun12/paddlevideo:tsn_dali_cuda9_0
```

基于以上docker镜像创建docker容器，运行命令为:

```bash
nvidia-docker run --name tsn-DALI -v /home:/workspace --network=host -it --shm-size 64g -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video huangjun12/paddlevideo:tsn_dali_cuda9_0 /bin/bash
```
- docker中安装好了飞桨2.0.0-rc1版本和我们二次开发后的DALI，创建容器后您可以在docker环境中直接开始tsn模型训练，无需额外配置环境。

## 数据准备

PaddleVide提供了在K400和UCF101两种数据集上训练TSN的训练脚本。

- K400数据下载及准备请参考[K400数据准备](../../dataset/k400.md)

- UCF101数据下载及准备请参考[UCF101数据准备](../../dataset/ucf101.md)

## 模型训练

### 预训练模型下载

加载在ImageNet1000上训练好的ResNet50权重作为Backbone初始化参数，请下载此[模型参数](https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams),
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

模型训练的启动命令为: 

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_tsn main.py --train_dali -c configs/recognition/tsn/tsn_dali.yaml -o log_level="INFO"
```

- 通过`-c`指定模型训练参数配置文件，模型及训练参数配置请参考配置文件```configs/recognition/tsn/tsn_dali.yaml```。

- 如若进行finetune，请下载PaddleVideo的已发布模型[comming soon]()， 通过`--weights`指定权重存放路径可进行模型finetune。 

- 您可以自定义修改参数配置，参数用法请参考[config](../../tutorials/config.md)。

## 模型测试

模型测试方法请参考TSN模型使用文档[模型测试部分](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/model_zoo/recognition/tsn.md#模型测试)

## 模型推理

模型推理方法请参考TSN模型使用文档[模型推理部分](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/model_zoo/recognition/tsn.md#模型推理)

## 参考论文

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool








