简体中文

# MoViNet视频分类模型

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [参考论文](#参考论文)

在开始使用之前，您可能需要按照以下命令安装额外的依赖包：
```bash
python -m pip install fvcore==0.1.5.post20210630
```

## 模型简介

MoViNet是Google Research研发的移动视频网络。它是一个可以用于在线推理视频流的，读取视频高效而且计算迅速的视频网络的一种。 这篇模型的产出还有EfficientNet一作的参与。
主要使用以下三步渐进式的方法来实现模型的搭建：

- 1 使用神经结构搜索的方法来搜索MoViNet空间结构
- 2 使用因果卷积算子和流缓冲区来弥补准确率的损失
- 3 Temporal Ensembles提升准确率 详细内容请参考CVPR
  2021年论文 [MoViNets: Mobile Video Networks for Efficient Video Recognition](https://arxiv.org/abs/2103.11511)

## 数据准备

UCF101数据下载及准备请参考[ucf101数据准备](../../dataset/ucf101.md)

## 模型训练

数据准备完成后，可通过如下方式启动训练：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_movinet main.py --validate -c configs/recognition/movinet/movinet_ucf101_frame.yaml
```

## 模型测试

- 若需单独运行测试代码，其启动命令如下：

```bash
python3.7 main.py --test -c configs/recognition/movinet/movinet_ucf101_frame.yaml -w output/MoViNet/MoViNet_best.pdparams
```

- 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。

## 参考论文

- [MoViNets: Mobile Video Networks for Efficient Video Recognition](https://arxiv.org/abs/2103.11511)
