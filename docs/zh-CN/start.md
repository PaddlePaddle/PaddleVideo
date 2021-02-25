简体中文 | [English](../en/getting_started.md)

# 开始使用
---
请参考[安装指南](./install.md)配置运行环境，并根据[数据](./dataset/)文档准备数据集，本章节下面所有的实验均以ucf101数据集为例。

PaddleVideo目前支持Linux下的GPU单卡和多卡运行环境。

- PaddleVideo各文件夹的默认存储路径， 以运行[example](../../configs/example.yaml)配置为例。

```
PaddleVideo
    ├── paddlevideo
    ├── ... #other source codes
    ├── output #ouput 权重，优化器参数等存储路径
    |    ├── example
    |    |   ├── example_best.pdparams #path_to_weights
    |    |   └── ...    
    |    └── ...    
    ├── log  #log存储路径
    |    ├── worker.0
    |    ├── worker.1
    |    └── ...    
    └── inference #预测文件存储路径
         ├── example.pdiparams file
         ├── example.pdimodel file
         └── example.pdiparmas.info file
```

<a name="1"></a>
## 1. 模型训练与评估

使用`paddle.distributed.launch`启动模型训练和测试脚本（`main.py`），可以更方便地启动多卡训练与测试，或直接运行(./run.sh)

```shell
sh run.sh
```
我们将所有标准的启动命令都放在了```run.sh```中，注意选择想要运行的脚本。

<a name="model_train"></a>
### 1.1 模型训练

参考如下方式启动模型训练，`paddle.distributed.launch`通过设置`gpus`指定GPU运行卡号，
指定`--validate`来启动训练时评估。

```bash
# PaddleVideo通过launch方式启动多卡多进程训练

export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
        --validate \
        -c ./configs/example.yaml
```

其中，`-c`用于指定配置文件的路径，可通过配置文件修改相关训练配置信息，也可以通过添加`-o`参数来更新配置：

```bash
python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
        -c ./configs/example.yaml \
        --validate \
        -o DATASET.batch_size=16
```
`-o`用于指定需要修改或者添加的参数，其中`-o DATASET.batch_size=16`表示更改batch_size大小为16。具体配置参数含义参考[配置文档](./tutorials/config.md#config-yaml-details)

运行上述命令，将会输出运行日志，并默认保存在./log目录下，如：`worker.0` , `worker.1` ... , worker日志文件对应每张卡上的输出

【train阶段】打印当前时间，当前epoch/epoch总数，当前batch id，评估指标，耗时，ips等信息：


    [12/28 17:31:26] epoch:[ 1/80 ] train step:0   loss: 0.04656 lr: 0.000100 top1: 1.00000 top5: 1.00000 elapse: 0.326 reader: 0.001s ips: 98.22489 instance/sec.


【eval阶段】打印当前时间，当前epoch/epoch总数，当前batch id，评估指标，耗时，ips等信息：


    [12/28 17:31:32] epoch:[ 80/80 ] val step:0    loss: 0.20538 top1: 0.88281 top5: 0.99219 elapse: 1.589 reader: 0.000s ips: 20.14003 instance/sec.


【epoch结束】打印当前时间，评估指标，耗时，ips等信息：


    [12/28 17:31:38] END epoch:80  val loss_avg: 0.52208 top1_avg: 0.84398 top5_avg: 0.97393 elapse_avg: 0.234 reader_avg: 0.000 elapse_sum: 7.021s ips: 136.73686 instance/sec.


当前为评估结果最好的epoch时，打印最优精度：

    [12/28 17:28:42] Already save the best model (top1 acc)0.8494


<a name="model_resume"></a>
### 1.2 模型恢复训练

如果训练任务终止，可以加载断点权重文件(优化器-学习率参数，断点文件)继续训练。
需要指定`-o resume_epoch`参数，该参数表示从```resume_epoch```轮开始继续训练.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
        -c ./configs/example.yaml \
        --validate \
        -o resume_epoch=5

```


<a name="model_finetune"></a>
### 1.3 模型微调

进行模型微调（Finetune），对自定义数据集进行模型微调，需要指定 `--weights` 参数来加载预训练模型。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
        -c ./configs/example.yaml \
        --validate \
        --weights=./output/example/path_to_weights
```

PaddleVideo会自动**不加载**shape不匹配的参数


<a name="model_test"></a>
### 1.4 模型测试

需要指定 `--test`来启动测试模式，并指定`--weights`来加载预训练模型。

```bash
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
        -c ./configs/example.yaml \
        --test \
        --weights=./output/example/path_to_weights
```



<a name="model_inference"></a>
## 2. 模型推理

通过导出inference模型，PaddlePaddle支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换
指定`-c`参数加载配置文件，指定`-p`参数加载模型权重，指定`-o`用于指定转换后模型的存储路径。

```bash
python tools/export_model.py \
    -c ./configs/example.yaml \
    -p ./output/example/path_to_weights \
    -o ./inference
```


上述命令将生成模型结构文件（`model_name.pdmodel`）和模型权重文件（`model_name.pdiparams`），然后可以使用预测引擎进行推理：

```bash
python tools/predict.py \
    --video_file "data/example.avi" \
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
