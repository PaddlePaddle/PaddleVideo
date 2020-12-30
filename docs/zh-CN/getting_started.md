# 开始使用
---
请参考[安装指南](./install.md)配置运行环境，并根据[数据](./data/ucf101)文档准备数据集，本章节下面所有的实验均以ucf101数据集为例。

PaddleVideo目前支持的训练/评估环境如下：
```shell
└── 单卡GPU
    └── Linux

└── 多卡GPU
    └── Linux
```

<a name="1"></a>
## 1. 模型训练与评估

使用`paddle.distributed.launch`启动模型训练脚本（`tools/train.py`）、测试脚本（`tools/test.py`），可以更方便地启动多卡训练与测试。

<a name="model_train"></a>
### 1.1 模型训练

参考如下方式启动模型训练，`paddle.distributed.launch`通过设置`gpus`指定GPU运行卡号：

```bash
# PaddleVideo通过launch方式启动多卡多进程训练

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/example.yaml
```

其中，`-c`用于指定配置文件的路径，可通过配置文件修改相关训练配置信息，也可以通过添加`-o`参数来更新配置：

```bash
python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/example.yaml \
        -o DATASET.batch_size=16
```
`-o`用于指定需要修改或者添加的参数，其中`-o DATASET.batch_size=16`表示更改batch_size大小为16。具体配置参数含义参考[配置文档](./config.md)

运行上述命令，将会输出运行日志，并默认保存在./log目录下，如：worker.0 , worker.1 ... , worker日志文件对应每张卡上的输出
train阶段，打印当前时间，当前epoch/总数，当前batch，评估指标，耗时，ips等信息：

    
    [12/28 17:31:26] epoch:[ 1/80 ] train step:0   loss: 0.04656 lr: 0.000100 top1: 1.00000 top5: 1.00000 elapse: 0.326 reader: 0.001s ips: 98.22489 instance/sec.
    
    
eval阶段，打印当前时间，当前epoch/总数，当前batch，评估指标，耗时，ips等信息：

    
    [12/28 17:31:32] epoch:[ 80/80 ] val step:0    loss: 0.20538 top1: 0.88281 top5: 0.99219 elapse: 1.589 reader: 0.000s ips: 20.14003 instance/sec.
    
    
训练每个epoch结束，打印当前时间，评估指标，耗时，ips等信息：

    
    [12/28 17:31:38] END epoch:80  val loss_avg: 0.52208 top1_avg: 0.84398 top5_avg: 0.97393 elapse_avg: 0.234 reader_avg: 0.000 elapse_sum: 7.021s ips: 136.73686 instance/sec.
    
    
当前为评估结果最好的epoch时，打印最优精度：

    
    [12/28 17:28:42] Already save the best model (top1 acc)0.8494
    

<a name="model_resume"></a>
### 1.2 模型恢复训练

如果训练任务因为其他原因被终止，也可以加载断点权重文件继续训练。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/example.yaml \
        -o resume_epoch=5

```

在继续训练时设置`resume_epoch`参数即可，该参数表示从```resume_epoch```轮开始继续训练，会同时加载保存的模型参数权重和学习率、优化器等信息。

<a name="model_test"></a>
### 1.3 模型测试

可以通过以下命令进行模型评估。

```bash
python tools/test.py \
    -c ./configs/example.yaml \
    -w ./outputs/path_to_weights
```



<a name="model_inference"></a>
## 2. 模型推理

通过导出inference模型，PaddlePaddle支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python tools/export_model.py \
    -c ./configs/example.yaml \
    -p ./output/path_to_weights \
    -o ./inference
```

其中，参数`-c`用于指定配置文件，`-p`用于指定模型权重路径，`--o`用于指定转换后模型的存储路径。

上述命令将生成模型结构文件（`model_name.pdmodel`）和模型权重文件（`model_name.pdiparams`），然后可以使用预测引擎进行推理：

```bash
python tools/infer/predict.py \
    --video_file 预测视频路径 \
    --model_file "./inference/example.pdmodel" \
    --params_file "./inference/example.pdiparams" \
    --use_gpu=True \
    --use_tensorrt=False
```
其中：
+ `video_file`：待预测的视频文件路径，如 `./test.avi`
+ `model_file`：模型结构文件路径，如 `./inference/example.pdmodel`
+ `params_file`：模型权重文件路径，如 `./inference/example.pdiparams`
+ `use_tensorrt`：是否使用 TesorRT 预测引擎，默认值：`True`
+ `use_gpu`：是否使用 GPU 预测，默认值：`True`

benchmark 预测速度结果由 `tools/predict.py` 进行评测 ，具体参考[评测文档](benchmark.md) 。
