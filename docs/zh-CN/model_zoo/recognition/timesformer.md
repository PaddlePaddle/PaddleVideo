[English](../../../en/model_zoo/recognition/timesformer.md) | 简体中文

# TimeSformer视频分类模型

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介

TimeSformer是基于vision transformer的视频分类模型，具有无卷积、全局感受野、时间序列建模能力强的特点。目前在Kinetics-400数据集上达到了SOTA精度，超过了经典的基于CNN的视频分类模型TSN和TSM以及Slowfast，而且具有更短的训练用时（Kinetics-400数据集训练用时39小时）。**本代码实现的是论文中的时间-空间分离的注意力级联网络**。

<div align="center">
<img src="../../../images/timesformer_attention_arch.png" alt="image-20210628210446041"/><img src="../../../images/timesformer_attention_visualize.png" alt="image-20210628210446041"  />
</div>


## 数据准备

K400数据下载及准备请参考[Kinetics-400数据准备](../../dataset/k400.md)

UCF101数据下载及准备请参考[UCF-101数据准备](../../dataset/ucf101.md)


## 模型训练

### Kinetics-400数据集训练

#### 下载并添加预训练模型

1. 下载图像预训练模型[ViT_base_patch16_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams)作为Backbone初始化参数，或通过wget命令下载

   ```bash
   wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams
   ```

2. 打开`PaddleVideo/configs/recognition/timesformer/timesformer_k400_videos.yaml`，将下载好的权重存放路径填写到下方`pretrained:`之后

    ```yaml
    MODEL:
        framework: "RecognizerTransformer"
        backbone:
            name: "VisionTransformer"
            pretrained: 将路径填写到此处
    ```

#### 开始训练

- Kinetics400数据集使用8卡训练，训练方式的启动命令如下:

    ```bash
    # videos数据格式
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_timesformer  main.py  --validate -c configs/recognition/timesformer/timesformer_k400_videos.yaml
    ```
    
- 开启amp混合精度训练，可加速训练过程，其训练启动命令如下：

    ```bash
    export FLAGS_conv_workspace_size_limit=800 # MB
    export FLAGS_cudnn_exhaustive_search=1
    export FLAGS_cudnn_batchnorm_spatial_persistent=1
    # videos数据格式
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_timesformer  main.py --amp --validate -c configs/recognition/timesformer/timesformer_k400_videos.yaml
    ```
    
- 另外您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的，建议配置文件的命名方式为`模型_数据集名称_文件格式_数据格式_采样方式.yaml`，参数用法请参考[config](../../tutorials/config.md)。


## 模型测试

- TimeSformer模型在训练时同步进行验证，您可以通过在训练日志中查找关键字`best`获取模型测试精度，日志示例如下:

  ```
  Already save the best model (top1 acc)0.7258
  ```

- 由于TimeSformer模型测试模式的采样方式是速度稍慢但精度高一些的**UniformCrop**，与训练过程中验证模式采用的**RandomCrop**不同，所以训练日志中记录的验证指标`topk Acc`不代表最终的测试分数，因此在训练完成之后可以用测试模式对最好的模型进行测试获取最终的指标，命令如下：

  ```bash
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_timesformer  main.py  --test -c configs/recognition/timesformer/timesformer_k400_videos.yaml -w "output/TimeSformer/TimeSformer_best.pdparams"
  ```


  当测试配置采用如下参数时，在Kinetics-400的validation数据集上的测试指标如下：

   |      backbone      | Sampling method | num_seg | target_size | Top-1 |                         checkpoints                          |
   | :----------------: | :-------------: | :-----: | :---------: | :---- | :----------------------------------------------------------: |
   | Vision Transformer |   UniformCrop   |   8    |     224     | 77.29 | [TimeSformer_k400.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TimeSformer_k400.pdparams) |


- 测试时，TimeSformer视频采样策略为使用Linspace采样：时序上，从待采样视频序列的第一帧到最后一帧区间内，均匀生成`num_seg`个稀疏采样点（包括端点）；空间上，选择长边两端及中间位置（左中右 或 上中下）3个区域采样。1个视频共采样1个clip。

## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/recognition/timesformer/timesformer_k400_videos.yaml \
                                -p data/TimeSformer_k400.pdparams \
                                -o inference/TimeSformer
```

上述命令将生成预测所需的模型结构文件`TimeSformer.pdmodel`和模型权重文件`TimeSformer.pdiparams`。

- 各参数含义可参考[模型推理方法](../../start.md#2-模型推理)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/timesformer/timesformer_k400_videos.yaml \
                           --model_file inference/TimeSformer/TimeSformer.pdmodel \
                           --params_file inference/TimeSformer/TimeSformer.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

输出示例如下:

```
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 0.9997474551200867
```

可以看到，使用在Kinetics-400上训练好的TimeSformer模型对`data/example.avi`进行预测，输出的top1类别id为`5`，置信度为0.99。通过查阅类别id与名称对应表`data/k400/Kinetics-400_label_list.txt`，可知预测类别名称为`archery`。

## 参考论文

- [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/pdf/2102.05095.pdf), Gedas Bertasius, Heng Wang, Lorenzo Torresani
