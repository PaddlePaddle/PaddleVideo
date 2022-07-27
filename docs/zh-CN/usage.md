简体中文 | [English](../en/start.md)

# 使用指南
---

* [1. 模型训练](#1)
* [2. 模型恢复训练](#2)
* [3. 模型微调](#3)
* [4. 模型测试](#4)
* [5. 模型推理](#5)
* [6. 混合精度训练](#6)


请参考[安装指南](./install.md)配置运行环境，PaddleVideo目前支持Linux下的GPU单卡和多卡运行环境。



<a name="1"></a>
## 1. 模型训练

PaddleVideo支持单机单卡和单机多卡训练，单卡训练和多卡训练的启动方式略有不同。

### 1.1 单卡训练

启动脚本示例:

```bash
export CUDA_VISIBLE_DEVICES=0         #指定使用的GPU显卡id
python3.7 main.py  --validate -c configs_path/your_config.yaml
```
- `-c` 必选参数，指定运行的配置文件路径，具体配置参数含义参考[配置文档](./contribute/config.md#config-yaml-details)
- `--validate` 可选参数，指定训练时是否评估
-  `-o`: 可选参数，指定重写参数，例如： `-o DATASET.batch_size=16` 用于重写train时batch size大小

### 1.2 多卡训练

通过`paddle.distributed.launch`启动，启动脚本示例:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=your_log_dir  main.py  --validate -c configs_path/your_config.yaml
```
- `--gpus`参数指定使用的GPU显卡id
- `--log_dir`参数指定日志保存目录
多卡训练详细说明可以参考[单机多卡训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.1/guides/02_paddle2.0_develop/06_device_cn.html#danjiduokaxunlian)


我们将所有标准的启动命令都放在了```run.sh```中，直接运行(./run.sh)可以方便地启动多卡训练与测试，注意选择想要运行的脚本
```shell
sh run.sh
```

### 1.3 输出日志

运行训练命令，将会输出运行日志，并默认保存在./log目录下，如：`worker.0` , `worker.1` ... , worker日志文件对应每张卡上的输出

【train阶段】打印当前时间，当前epoch/epoch总数，当前batch id，评估指标，耗时，ips等信息：
```txt
[09/24 14:13:00] epoch:[  1/1  ] train step:100  loss: 5.31382 lr: 0.000250 top1: 0.00000 top5: 0.00000 batch_cost: 0.73082 sec, reader_cost: 0.38075 sec, ips: 5.47330 instance/sec.
```

【eval阶段】打印当前时间，当前epoch/epoch总数，当前batch id，评估指标，耗时，ips等信息：
```txt
[09/24 14:16:55] epoch:[  1/1  ] val step:0    loss: 4.42741 top1: 0.00000 top5: 0.00000 batch_cost: 1.37882 sec, reader_cost: 0.00000 sec, ips: 2.90104 instance/sec.
```

【epoch结束】打印当前时间，评估指标，耗时，ips等信息：
```txt
[09/24 14:18:46] END epoch:1   val loss_avg: 5.21620 top1_avg: 0.02215 top5_avg: 0.08808 avg_batch_cost: 0.04321 sec, avg_reader_cost: 0.00000 sec, batch_cost_sum: 112.69575 sec, avg_ips: 8.41203 instance/sec.
```

当前为评估结果最好的epoch时，打印最优精度：
```txt
[09/24 14:18:47] Already save the best model (top1 acc)0.7467
```

### 1.4 输出存储路径

- PaddleVideo各文件夹的默认存储路径如下：

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

- 训练Epoch默认从1开始计数，参数文件的保存格式为`ModelName_epoch_00001.pdparams`，命名中的数字对应Epoch编号。


<a name="2"></a>

## 2. 模型恢复训练

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

<a name="3"></a>

## 3. 模型微调

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


<a name="4"></a>

## 4. 模型测试

需要指定 `--test`来启动测试模式，并指定`--weights`来加载预训练模型。

```bash
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    main.py \
    -c ./configs/example.yaml \
    --test \
    --weights=./output/example/path_to_weights
```

<a name="5"></a>

## 5. 模型推理

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
    --input_file "data/example.avi" \
    --model_file "./inference/TSN.pdmodel" \
    --params_file "./inference/TSN.pdiparams" \
    --use_gpu=True \
    --use_tensorrt=False
```

其中：

+ `input_file`：待预测的文件路径或文件夹路径，如 `./test.avi`
+ `model_file`：模型结构文件路径，如 `./inference/TSN.pdmodel`
+ `params_file`：模型权重文件路径，如 `./inference/TSN.pdiparams`
+ `use_tensorrt`：是否使用 TesorRT 预测引擎，默认值：`False`
+ `use_gpu`：是否使用 GPU 预测，默认值：`True`


<a name="6"></a>

## 6. 混合精度训练

[混合精度训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/performance_improving/amp_cn.html#amp)使用fp16数据类型进行训练，可以加速训练过程，减少显存占用，其训练启动命令如下：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export FLAGS_conv_workspace_size_limit=800 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=your_log_dir  main.py --amp --validate -c configs_path/your_config.yaml
```

各模型详细的使用文档，可以参考[Models](./model_zoo/README.md)
