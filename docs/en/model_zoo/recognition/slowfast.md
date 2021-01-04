# SlowFast

---
## Content

- [Introduction](#Introduction)
- [Data Preparation](#Data Preparation)
- [Train](#Train)
- [Test](#Test)
- [Reference](#Reference)


## Introduction

SlowFast是视频分类领域的高精度模型，使用slow和fast两个分支。slow分支以稀疏采样得到的帧作为输入，捕捉视频中的表观信息。fast分支以高频采样得到的帧作为输入，捕获视频中的运动信息，最终将两个分支的特征拼接得到预测结果。

<p align="center">
<img src="./SLOWFAST.png" height=300 width=500 hspace='10'/> <br />
SlowFast Overview
</p>

详细内容请参考ICCV 2019论文[SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)


## Data Preparation

SlowFast模型的训练数据采用Kinetics400数据集，数据下载及准备请参考[数据说明](../PaddleCV/video/data/dataset/README.md)


## Train

数据准备完成后，可通过如下两种方式启动训练：

默认使用8卡训练，启动方式如下:

    bash run_train_multi.sh

若使用单卡训练，启动方式如下:

    bash run_train_single.sh

- 建议使用多卡训练方式，单卡由于batch\_size减小，精度可能会有损失。

- 从头开始训练，使用上述启动命令行或者脚本程序即可启动训练，不需要用到预训练模型。

- Visual DL可以用来对训练过程进行可视化，具体使用方法请参考[VisualDL](https://github.com/PaddlePaddle/VisualDL)

**训练资源要求：**

*  8卡V100，总batch\_size=64，单卡batch\_size=8，单卡显存占用约9G。
*  Kinetics400训练集较大(约23万个样本)，SlowFast模型迭代epoch数较多(196个)，因此模型训练耗时较长，约200个小时。
*  训练加速工作进行中，敬请期待。


## Test

训练完成后，可通过如下方式进行模型评估:

多卡评估方式如下：

    bash run_eval_multi.sh

若使用单卡评估，启动方式如下：

    bash run_eval_single.sh

- 进行评估时，可修改脚本中的`weights`参数指定用到的权重文件，如果不设置，将使用默认参数文件checkpoints/slowfast_epoch195.pdparams。

- 使用```multi_crop```的方式进行评估，因此评估有一定耗时，建议使用多卡评估，加快评估速度。若使用默认方式进行多卡评估，耗时约4小时。

- 模型最终的评估精度会打印在日志文件中。


在Kinetics400数据集下评估精度如下:

| Acc1 | Acc5 |
| :---: | :---: |
| 74.35 | 91.33 |

- 由于Kinetics400数据集部分源文件已缺失，无法下载，我们使用的数据集比官方数据少~5%，因此精度相比于论文公布的结果有一定损失。


## Reference

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)
