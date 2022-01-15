# 视频质量评价模型
---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型优化](#模型优化)
- [模型部署](#模型部署)
- [参考论文](#参考论文)


## 模型简介

该代码库主要基于paddle2.1版本开发，主要是在ppTSM网络模型的基础上修改的一种无参考视频质量评估方法，通过读入视频的视频帧来判断该视频的质量。

针对视频内容的理解，可以自动分析视频内容的质量，帮助选出最优的关键帧或关键片段作为视频封面，提升视频的点击转换和用户体验。

本项目目前支持Linux下的GPU单卡和多卡运行环境。

## 数据准备

```
数据集来自公开数据集KonVid-150k，共153842个ugc视频，其中训练集(KonVid-150k-A)152265个，验证集(KonVid-150k-B)1577个
示例数据集以及数据集官网地址: datasets/dataset_url.list
数据集标注文件为dataset中的train.txt和eval.txt
```

## 模型训练

环境安装：

- PaddlePaddle >= 2.1.0
- Python >= 3.7
- PaddleX >= 2.0.0

- CUDA >= 10.1
- cuDNN >= 7.6.4
- nccl >= 2.1.2

安装Python依赖库：

Python依赖库在[requirements.txt](https://github.com/PaddlePaddle/PaddleVideo/blob/master/requirements.txt)中给出，可通过如下命令安装：

```
python3.7 -m pip install --upgrade pip
pip3.7 install --upgrade -r requirements.txt
```

使用`paddle.distributed.launch`启动模型训练和测试脚本（`main.py`），可以更方便地启动多卡训练与测试，或直接运行(./run.sh)

```shell
sh run.sh
```
我们将所有标准的启动命令都放在了```run.sh```中，注意选择想要运行的脚本。

参考如下方式启动模型训练，`paddle.distributed.launch`通过设置`gpus`指定GPU运行卡号，
指定`--validate`来启动训练时评估。

```bash
# PaddleVideo通过launch方式启动多卡多进程训练

export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    --log_dir=log_pptsm \
    main.py \
    --amp \
    --validate \
    -c ./configs/recognition/tsm/pptsm_regression.yaml
```

其中，`-c`用于指定配置文件的路径，可通过配置文件修改相关训练配置信息，也可以通过添加`-o`参数来更新配置：

```bash
python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
    -c ./configs/recognition/tsm/pptsm_regression.yaml \
    --validate \
    -o DATASET.batch_size=16
```
`-o`用于指定需要修改或者添加的参数，其中`-o DATASET.batch_size=16`表示更改batch_size大小为16。

运行上述命令，将会输出运行日志，并默认保存在./log目录下，如：`worker.0` , `worker.1` ... , worker日志文件对应每张卡上的输出

【train阶段】打印当前时间，当前epoch/epoch总数，当前batch id，评估指标，耗时，ips等信息：


    [11/16 04:40:37] epoch:[  1/1  ] train step:100  loss: 5.31382 lr: 0.000250 batch_cost: 0.73082 sec, reader_cost: 0.38075 sec, ips: 5.47330 instance/sec.


【eval阶段】打印当前时间，当前epoch/epoch总数，当前batch id，评估指标，耗时，ips等信息：


    [11/16 04:40:37] epoch:[  1/1  ] val step:0    loss: 4.42741 batch_cost: 1.37882 sec, reader_cost: 0.00000 sec, ips: 2.90104 instance/sec.


【epoch结束】打印当前时间，学习率，评估指标，耗时，ips等信息：


    [11/16 04:40:37] lr=0.00012487
    [11/16 04:40:37] train_SROCC=0.4456697876616565
    [11/16 04:40:37] train_PLCC=0.48071880604403616
    [11/16 04:40:37] END epoch:1   val loss_avg: 5.21620 avg_batch_cost: 0.04321 sec, avg_reader_cost: 0.00000 sec, batch_cost_sum: 112.69575 sec, avg_ips: 8.41203 instance/sec.


当前为评估结果最好的epoch时，打印最优精度：

    [11/16 04:40:57] max_SROCC=0.7116468111328617
    [11/16 04:40:57] max_PLCC=0.733503995526737

### 模型恢复训练

如果训练任务终止，可以加载断点权重文件(优化器-学习率参数，断点文件)继续训练。
需要指定`-o resume_epoch`参数，该参数表示从```resume_epoch```轮开始继续训练.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
    --amp \
    -c ./configs/recognition/tsm/pptsm_regression.yaml \
    --validate \
    -o resume_epoch=5

```

### 模型微调

进行模型微调（Finetune），对自定义数据集进行模型微调，需要指定 `--weights` 参数来加载预训练模型。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
    --amp \
    -c ./configs/recognition/tsm/pptsm_regression.yaml \
    --validate \
    --weights=./output/model_name/ppTSM_best.pdparams
```

PaddleVideo会自动**不加载**shape不匹配的参数

## 模型测试

需要指定 `--test`来启动测试模式，并指定`--weights`来加载预训练模型。

```bash
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
    -c ./configs/recognition/tsm/pptsm_regression.yaml \
    --test \
    --weights=./output/model_name/ppTSM_best.pdparams
```

## 模型优化

在实际使用场景中可根据视频质量以及尺寸尝试优化策略

- 可通过原图输入来替换RandomCrop:224操作，准确率由SROCC=0.8176,PLCC=0.8361提升到SROCC=0.8617,PLCC=0.8910,不同模型以及特征增强操作的效果对比如下表所示

  |  模型  |                  特征增强                   | val_SROCC | val_PLCC |
  | :----: | :-----------------------------------------: | :-------: | :------: |
  | GSTVQA |                  原图输入                   |  0.7932   |  0.8006  |
  | ppTSM  | train--RandomCrop=224  val--center_crop=224 |  0.8176   |  0.8361  |
  | ppTSM  | train--RandomCrop=512  val--center_crop=512 |  0.8603   |  0.8822  |
  | ppTSM  |                  原图输入                   |  0.8617   |  0.8910  |

  

- 考虑应用场景视频的 aspect ratio 大都为 16：9 和 4：3 等，同时为了避免非均匀缩放拉伸带来的干扰 ，可以采用了（224x3）x(224x2)=672x448 的输入尺寸来更充分得利用有限的输入尺寸。 

## 模型部署

本代码解决方案在官方验证集(KonVid-150k-B)上的指标效果为SROCC=0.8176,PLCC=0.8361。

## 参考论文

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han

- [ [Quality Assessment of In-the-Wild Videos](https://dl.acm.org/citation.cfm?doid=3343031.3351028)](https://arxiv.org/pdf/1811.08383.pdf), Dingquan Li, Tingting Jiang, and Ming Jiang

