[English](../../../en/model_zoo/recognition/movinet.md) | 简体中文

# MoViNet视频分类模型

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介

MoViNet是Google Research研发的移动视频网络。它使用神经结构搜索的方法来搜索MoViNet空间结构，使用因果卷积算子和流缓冲区来弥补准确率的损失，Temporal Ensembles提升准确率，是一个可以用于在线推理视频流的，轻量高效视频模型。

## 数据准备

Kinetics-400数据下载及准备请参考[kinetics-400数据准备](../../dataset/k400.md)

## 模型训练

数据准备完成后，可通过如下方式启动训练：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_movinet main.py --validate -c configs/recognition/movinet/movinet_k400_frame.yaml
```

## 模型测试

- MoViNet模型在训练时同步进行测试，您可以通过在训练日志中查找关键字`best`获取模型测试精度，日志示例如下:

```txt
Already save the best model (top1 acc)0.6489
```

- 若需单独运行测试代码，其启动命令如下：

```bash
python3.7 main.py --test -c configs/recognition/movinet/movinet_k400_frame.yaml -w output/MoViNet/MoViNet_best.pdparams
```

- 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。

当测试配置采用如下参数时，在Kinetics-400的validation数据集上的评估精度如下：

| Config | Sampling method | num_seg | target_size | Top-1 | checkpoints |
| :------: | :--------: | :-------: | :-------: | :-----: | :-----: |
| A0 | Uniform | 50 | 172  | 66.62 | [MoViNetA0_k400.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/MoViNetA0_k400.pdparams)  |


## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/recognition/movinet/movinet_k400_frame.yaml \
                                -p data/MoViNetA0_k400.pdparams \
                                -o inference/MoViNetA0
```

上述命令将生成预测所需的模型结构文件`MoViNetA0.pdmodel`和模型权重文件`MoViNetA0.pdiparams`。

各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/movinet/movinet_k400_frame.yaml \
                           --model_file inference/MoViNetA0/MoViNet.pdmodel \
                           --params_file inference/MoViNetA0/MoViNet.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

输出示例如下:
```txt
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 0.7667049765586853
```

## 参考论文

- [MoViNets: Mobile Video Networks for Efficient Video Recognition](https://arxiv.org/abs/2103.11511)
