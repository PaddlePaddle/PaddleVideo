[English](../../../en/model_zoo/recognition/tokenshift_transformer.md) | 简体中文

# Token Shift Transformer视频分类模型

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介

Token Shift Transformer 是基于vision transformer的视频分类模型，具有可解释性强、对超大规模数据具有高判别能力以及处理不同长度输入的灵活性等优点。Token Shift Module 是一种新颖的零参数、零 FLOPs 模块，用于对每个 Transformer编码器内的时间关系进行建模。

<div align="center">
<img src="../../../images/tokenshift_structure.png">
</div>


## 数据准备

UCF-101数据下载及准备请参考[UCF-101数据准备](../../dataset/ucf101.md)


## 模型训练

### UCF-101数据集训练

#### 下载并添加预训练模型

1. 下载图像预训练模型[ViT_base_patch16_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams)作为Backbone初始化参数，或通过wget命令下载。

   ```bash
   wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams
   ```

2. 打开`PaddleVideo/configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml`，将下载好的权重存放路径填写到下方`pretrained:`之后

    ```yaml
    MODEL:
        framework: "RecognizerTransformer"
        backbone:
            name: "TokenShiftVisionTransformer"
            pretrained: 将路径填写到此处
    ```

#### 开始训练

- UCF-101数据集使用单卡训练，训练方式的启动命令如下:

    ```bash
    # videos数据格式
    python3 main.py -c configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml --validate --seed=1234
    ```
    
- 开启amp混合精度训练，可加速训练过程，其训练启动命令如下：

    ```bash
    # videos数据格式
    python3 main.py --amp -c configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml --validate --seed=1234
    ```
    
- 另外您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的，建议配置文件的命名方式为`模型_数据集名称_文件格式_数据格式_采样方式.yaml`，参数用法请参考[config](../../contribute/config.md)。


## 模型测试

- Token Shift Transformer模型在训练时同步进行验证，您可以通过在训练日志中查找关键字`best`获取模型测试精度，日志示例如下:

  ```
  Already save the best model (top1 acc)0.9201
  ```

- 由于Token Shift Transformer模型测试模式的采样方为的**uniform**采样，与训练过程中验证模式采用的**dense**采样不同，所以训练日志中记录的验证指标`topk Acc`不代表最终的测试分数，因此在训练完成之后可以用测试模式对最好的模型进行测试获取最终的指标，命令如下：

  ```bash
  python3 main.py --amp -c configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml --test --seed=1234 -w 'output/TokenShiftVisionTransformer/TokenShiftVisionTransformer_best.pdparams'
  ```


  当测试配置采用如下参数时，在UCF-101的validation数据集上的测试指标如下：

   |      backbone      | sampling method | num_seg | target_size | Top-1 |                         checkpoints                          |
   | :----------------: | :-------------: | :-----: | :---------: | :---- | :----------------------------------------------------------: |
  | Vision Transformer | Uniform | 8 | 256 | 92.81 | [TokenShiftTransformer.pdparams](https://drive.google.com/drive/folders/1k_TpAqaJZYJE8C5g5pT9phdyk9DrY_XL?usp=sharing) |


- Uniform采样: 时序上，等分成`num_seg`段，每段中间位置采样1帧；空间上，中心位置采样。1个视频共采样1个clip。

## 模型推理

### 导出inference模型

```bash
python3 tools/export_model.py -c configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml -p 'output/TokenShiftVisionTransformer/TokenShiftVisionTransformer_best.pdparams'
```

上述命令将生成预测所需的模型结构文件`TokenShiftVisionTransformer.pdmodel`和模型权重文件`TokenShiftVisionTransformer.pdiparams`。

- 各参数含义可参考[模型推理方法](../../usage.md#5-模型推理)

### 使用预测引擎推理

```bash
python3 tools/predict.py -c configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml -i 'data/BrushingTeeth.avi' --model_file ./inference/TokenShiftVisionTransformer.pdmodel --params_file ./inference/TokenShiftVisionTransformer.pdiparams
```

输出示例如下:

```
Current video file: data/BrushingTeeth.avi
	top-1 class: 19
	top-1 score: 0.9959074258804321
```

可以看到，使用在UCF-101上训练好的Token Shift Transformer模型对`data/BrushingTeeth.avi`进行预测，输出的top1类别id为`19`，置信度为0.99。通过查阅类别id与名称对应表，可知预测类别名称为`brushing_teeth`。

## 参考论文

- [Token Shift Transformer for Video Classification](https://arxiv.org/pdf/2108.02432v1.pdf), Zhang H, Hao Y, Ngo C W.
