[English](../../../en/model_zoo/recognition/pp-tsn.md) | 简体中文

# PP-TSN视频分类模型

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介

我们对[TSN模型](./tsn.md)进行了改进，得到了更高精度的2D实用视频分类模型**PP-TSN**。在不增加参数量和计算量的情况下，在UCF-101、Kinetics-400等数据集上精度显著超过原版，在Kinetics-400数据集上的精度如下表所示。

| Version | Top1 |
| :------ | :----: |
| Ours (distill) | 75.06 |
| Ours | **73.68** |
| [mmaction2](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn#kinetics-400) | 71.80 |


## 数据准备

K400数据下载及准备请参考[Kinetics-400数据准备](../../dataset/k400.md)

UCF101数据下载及准备请参考[UCF-101数据准备](../../dataset/ucf101.md)


## 模型训练

### Kinetics-400数据集训练

#### 下载并添加预训练模型

1. 下载图像蒸馏预训练模型[ResNet50_vd_ssld_v2.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams)作为Backbone初始化参数，或通过wget命令下载

   ```bash
   wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams
   ```

2. 打开`PaddleVideo/configs/recognition/pptsn/pptsn_k400_frames.yaml`，将下载好的权重存放路径填写到下方`pretrained:`之后

    ```yaml
    MODEL:
        framework: "Recognizer2D"
        backbone:
            name: "ResNetTweaksTSN"
            pretrained: 将路径填写到此处
    ```

#### 开始训练

- Kinetics400数据集使用8卡训练，训练方式的启动命令如下:

    ```bash
    # frames数据格式
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsn  main.py  --validate -c configs/recognition/pptsn/pptsn_k400_frames.yaml

    # videos数据格式
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsn  main.py  --validate -c configs/recognition/pptsn/pptsn_k400_videos.yaml
    ```

- 开启amp混合精度训练，可加速训练过程，其训练启动命令如下：

    ```bash
    export FLAGS_conv_workspace_size_limit=800 # MB
    export FLAGS_cudnn_exhaustive_search=1
    export FLAGS_cudnn_batchnorm_spatial_persistent=1

    # frames数据格式
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsn  main.py --amp --validate -c configs/recognition/pptsn/pptsn_k400_frames.yaml

    # videos数据格式
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsn  main.py --amp --validate -c configs/recognition/pptsn/pptsn_k400_videos.yaml
    ```

- 另外您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的，建议配置文件的命名方式为`模型_数据集名称_文件格式_数据格式_采样方式.yaml`，参数用法请参考[config](../../tutorials/config.md)。


## 模型测试

- PP-TSN模型在训练时同步进行验证，您可以通过在训练日志中查找关键字`best`获取模型测试精度，日志示例如下:

  ```
  Already save the best model (top1 acc)0.7004
  ```

- 由于PP-TSN模型测试模式的采样方式是速度稍慢但精度高一些的**TenCrop**，与训练过程中验证模式采用的**CenterCrop**不同，所以训练日志中记录的验证指标`topk Acc`不代表最终的测试分数，因此在训练完成之后可以用测试模式对最好的模型进行测试获取最终的指标，命令如下：

  ```bash
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_pptsn main.py  --test -c configs/recognition/pptsn/pptsn_k400_frames.yaml -w "output/ppTSN/ppTSN_best.pdparams"
  ```


  当测试配置采用如下参数时，在Kinetics-400的validation数据集上的测试指标如下：


  | backbone | Sampling method | distill | num_seg | target_size | Top-1 | checkpoints |
  | :------: | :----------: | :----: | :----: | :----: | :---- | :---: |
  | ResNet50 | TenCrop | False | 3 | 224 | 73.68 | [ppTSN_k400.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSN_k400.pdparams) |
  | ResNet50 | TenCrop | True | 8 | 224 | 75.06 | [ppTSN_k400_8.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSN_k400_8.pdparams) |

- PP-TSN视频采样策略为TenCrop采样：时序上，将待输入视频均匀分成`num_seg`段区间，每段的中间位置采样1帧；空间上，从左上角、右上角、中心点、左下角、右下角5个子区域各采样224x224的区域，并加上水平翻转，一共得到10个采样结果。1个视频共采样1个clip。

- distill为`True`表示使用了蒸馏所得的预训练模型，具体蒸馏方案参考[ppTSM蒸馏方案](TODO)。


## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/recognition/pptsn/pptsn_k400_frames.yaml -p data/ppTSN_k400.pdparams -o inference/ppTSN
```

上述命令将生成预测所需的模型结构文件`ppTSN.pdmodel`和模型权重文件`ppTSN.pdiparams`以及`ppTSN.pdiparams.info`文件，均存放在`inference/ppTSN/`目录下

上述bash命令中各个参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/pptsn/pptsn_k400_frames.yaml \
                           --model_file inference/ppTSN/ppTSN.pdmodel \
                           --params_file inference/ppTSN/ppTSN.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

输出示例如下:

```bash
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 0.998979389667511
```

可以看到，使用在Kinetics-400上训练好的PP-TSN模型对`data/example.avi`进行预测，输出的top1类别id为`5`，置信度为0.99。通过查阅类别id与名称对应表`data/k400/Kinetics-400_label_list.txt`，可知预测类别名称为`archery`。
- **注意**：对于在计算时会合并N和T的模型（比如TSN、TSM），当`use_tensorrt=True`时，需要指定`batch_size`参数为batch_size*num_seg。

    ```bash
    python3.7 tools/predict.py --input_file data/example.avi \
                               --config configs/recognition/pptsn/pptsn_k400_frames.yaml \
                               --model_file inference/ppTSN/ppTSN.pdmodel \
                               --params_file inference/ppTSN/ppTSN.pdiparams \
                               --batch_size 8 \
                               --use_gpu=True \
                               --use_tensorrt=True
    ```
## 参考论文

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/pdf/1608.00859.pdf), Limin Wang, Yuanjun Xiong, Zhe Wang
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), Geoffrey Hinton, Oriol Vinyals, Jeff Dean
