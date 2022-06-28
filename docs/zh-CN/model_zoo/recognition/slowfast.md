简体中文 | [English](../../../en/model_zoo/recognition/slowfast.md)

# SlowFast视频分类模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介

SlowFast是视频分类领域的高精度模型，使用slow和fast两个分支。slow分支以稀疏采样得到的帧作为输入，捕捉视频中的表观信息。fast分支以高频采样得到的帧作为输入，捕获视频中的运动信息，最终将两个分支的特征拼接得到预测结果。

<p align="center">
<img src="../../../images/SlowFast.png" height=300 width=500 hspace='10'/> <br />
SlowFast Overview
</p>

详细内容请参考ICCV 2019论文[SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)


## 数据准备

SlowFast模型的训练数据采用Kinetics400数据集，数据下载及准备请参考[Kinetics-400数据准备](../../dataset/k400.md)


## 模型训练

数据准备完成后，可通过如下方式启动训练：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_slowfast  main.py --validate -c configs/recognition/slowfast/slowfast.yaml
```

- 从头开始训练，使用上述启动命令行或者脚本程序即可启动训练，不需要用到预训练模型。

- 建议使用多卡训练方式，单卡由于batch\_size减小，精度可能会有损失。


### 训练资源要求

*  8卡V100，总batch\_size=64，单卡batch\_size=8，单卡显存占用约9G。
*  训练速度相较原始实现提速100%，详细参考[benchmark](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/benchmark.md#实验结果)

### 训练加速

SlowFast为3D模型，训练异常耗时，为进一步加速模型的训练，我们实现了[Multigrid加速策略算法](https://arxiv.org/abs/1912.00998)，其训练启动方式如下:

```bash
python -B -m paddle.distributed.launch --selected_gpus="0,1,2,3,4,5,6,7" --log_dir=log-slowfast main.py --validate --multigrid -c configs/recognition/slowfast/slowfast_multigrid.yaml
```

性能数据如下:

| 训练策略 | 单个epoch平均耗时/min | 训练总时间/min | 加速比 |
| :------ | :-----: | :------: |:------: |
| Multigrid | 27.25 |  9758(6.7天) | 2.89x |
| Normal | 78.76 | 15438(10.7天) | base |

速度详细数据说明可参考[加速文档](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/tutorials/accelerate.md#%E8%AE%AD%E7%BB%83%E7%AD%96%E7%95%A5%E5%8A%A0%E9%80%9F)。

## 模型测试

可通过如下命令进行模型测试:

```bash
python -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_slowfast_test main.py --test -c  configs/recognition/slowfast/slowfast.yaml -w output/SlowFast/SlowFast_epoch_000196.pdparams
```

- 通过 `-w`参数指定待测试模型文件的路径，您可以下载我们训练好的模型进行测试[SlowFast.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast.pdparams)

- 使用```multi_crop```的方式进行评估，因此评估有一定耗时，建议使用多卡评估，加快评估速度。若使用默认方式进行多卡评估，耗时约4小时。

- 模型最终的评估精度会打印在日志文件中。

若使用单卡评估，启动方式如下：

```bash
python -B main.py --test -c  configs/recognition/slowfast/slowfast.yaml -w output/SlowFast/SlowFast_epoch_000196.pdparams
```


在Kinetics400数据集下评估精度及权重文件如下:

| Configs | Acc1 | Acc5 | Weights |
| :---: | :---: | :---: | :---: |
|  [slowfast.yaml](../../../../configs/recognition/slowfast/slowfast.yaml) | 74.35 | 91.33 | [slowfast_4x16.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast.pdparams) |
|  [slowfast_multigrid.yaml](../../../../configs/recognition/slowfast/slowfast_multigrid.yaml) | 75.84  | 92.33 | [slowfast_8x8.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast_8*8.pdparams) |

- 由于Kinetics400数据集部分源文件已缺失，无法下载，我们使用的数据集比官方数据少~5%，因此精度相比于论文公布的结果有一定损失。相同数据下，精度已与原实现对齐。


## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/recognition/slowfast/slowfast.yaml \
                                -p data/SlowFast.pdparams \
                                -o inference/SlowFast
```

上述命令将生成预测所需的模型结构文件`SlowFast.pdmodel`和模型权重文件`SlowFast.pdiparams`。

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/slowfast/slowfast.yaml \
                           --model_file inference/SlowFast/SlowFast.pdmodel \
                           --params_file inference/SlowFast/SlowFast.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

输出示例如下:

```
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 1.0
```

可以看到，使用在Kinetics-400上训练好的SlowFast模型对`data/example.avi`进行预测，输出的top1类别id为`5`，置信度为1.0。通过查阅类别id与名称对应表`data/k400/Kinetics-400_label_list.txt`，可知预测类别名称为`archery`。


## 参考论文

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982), Feichtenhofer C, Fan H, Malik J, et al.
- [A Multigrid Method for Efficiently Training Video Models](https://arxiv.org/abs/1912.00998), Chao-Yuan Wu, Ross Girshick, et al.
