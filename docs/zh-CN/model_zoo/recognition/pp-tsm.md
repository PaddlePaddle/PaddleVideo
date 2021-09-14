[English](../../../en/model_zoo/recognition/pp-tsm.md) | 简体中文

# PP-TSM视频分类模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介

我们对[TSM模型](./tsm.md)进行了改进，提出了高精度2D实用视频分类模型**PP-TSM**。在不增加参数量和计算量的情况下，在UCF-101、Kinetics-400等数据集上精度显著超过原文，在Kinetics-400数据集上的精度如下表所示。模型优化解析请参考[**PP-TSM模型精度优化Tricks详解**](https://zhuanlan.zhihu.com/p/382134297)。

| Version | Sampling method | Top1 |
| :------ | :----------: | :----: |
| Ours (distill) | Dense | **76.16** |
| Ours | Dense | 75.69 |
| [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README.md) | Dense | 74.55 |
| [mit-han-lab](https://github.com/mit-han-lab/temporal-shift-module) | Dense | 74.1 |

| Version | Sampling method | Top1 |
| :------ | :----------: | :----: |
| Ours (distill) | Uniform | **75.11** |
| Ours | Uniform | 74.54 |
| [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README.md) |  Uniform | 71.90 |
| [mit-han-lab](https://github.com/mit-han-lab/temporal-shift-module)  | Uniform | 71.16 |


## 数据准备

K400数据下载及准备请参考[Kinetics-400数据准备](../../dataset/k400.md)

UCF101数据下载及准备请参考[UCF-101数据准备](../../dataset/ucf101.md)


## 模型训练

### Kinetics-400数据集训练

#### 下载并添加预训练模型

下载图像蒸馏预训练模型[ResNet50_vd_ssld_v2.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams)作为Backbone初始化参数，或是通过命令行下载

```bash
wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams
```

并将文件路径添加到配置文件中的`MODEL.framework.backbone.pretrained`字段，如下：

```yaml
MODEL:
    framework: "Recognizer2D"
    backbone:
        name: "ResNetTweaksTSM"
        pretrained: 将路径填写到此处
```

#### 开始训练

- Kinetics400数据集使用8卡训练，frames格式数据，uniform训练方式的启动命令如下:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml
```

- Kinetics400数据集使用8卡训练，videos格式数据，uniform训练方式的启动命令如下:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/pptsm/pptsm_k400_videos_uniform.yaml
```

- 开启amp混合精度训练，可加速训练过程，其训练启动命令如下：

```bash
export FLAGS_conv_workspace_size_limit=800 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --amp --validate -c configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml
```

- Kinetics400数据集frames格式数据，dense训练方式的启动命令如下:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/pptsm/pptsm_k400_frames_dense.yaml
```

- 另外您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的，配置文件命名方式为`模型_数据集_文件格式_数据格式_采样方式.yaml`，参数用法请参考[config](../../tutorials/config.md)。


## 模型测试

- 对Uniform采样方式，PP-TSM模型在训练时同步进行测试，您可以通过在训练日志中查找关键字`best`获取模型测试精度，日志示例如下:

```txt
Already save the best model (top1 acc)0.7454
```

- 对dense采样方式，需单独运行测试代码，其启动命令如下：

```bash
python3 main.py --test -c configs/recognition/pptsm/pptsm_k400_frames_dense.yaml -w output/ppTSM/ppTSM_best.pdparams
```

- 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。


Kinetics400数据集测试精度:

| backbone | distill | Sampling method | num_seg | target_size | Top-1 | checkpoints |
| :------: | :----------: | :----: | :----: | :----: | :----: | :---- |
| ResNet50 | False | Uniform | 8 | 224 | 74.54 | [ppTSM_k400_uniform.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_uniform.pdparams) |
| ResNet50 | False | Dense | 8 | 224 | 75.69 | [ppTSM_k400_dense.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense.pdparams) |
| ResNet50 | True | Uniform | 8 | 224 | 75.11 | [ppTSM_k400_uniform_distill.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_uniform_distill.pdparams) |
| ResNet50 | True | Dense | 8 | 224 | 76.16 | [ppTSM_k400_dense_distill.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense_distill.pdparams) |

- Uniform采样: 时序上，等分成`num_seg`段，每段中间位置采样1帧；空间上，中心位置采样。1个视频共采样1个clips。

- Dense采样：时序上，先等分成10个片段，每段从起始位置开始，以`64//num_seg`为间隔连续采样`num_seg`帧；空间上，左中，中心，右中3个位置采样。1个视频共采样`10*3=30`个clips。

- distill为`True`表示使用了蒸馏所得的预训练模型，具体蒸馏方案参考[ppTSM蒸馏方案](TODO)。


## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml \
                                -p data/ppTSM_k400_uniform.pdparams \
                                -o inference/ppTSM
```

上述命令将生成预测所需的模型结构文件`ppTSM.pdmodel`和模型权重文件`ppTSM.pdiparams`。

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml \
                           --model_file inference/ppTSM/ppTSM.pdmodel \
                           --params_file inference/ppTSM/ppTSM.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```


输出示例如下:

```
Current video file: data/example.avi
	top-1 class: 5
	top-1 score: 0.9907386302947998
```


可以看到，使用在Kinetics-400上训练好的PP-TSM模型对`data/example.avi`进行预测，输出的top1类别id为`5`，置信度为0.99。通过查阅类别id与名称对应表`data/k400/Kinetics-400_label_list.txt`，可知预测类别名称为`archery`。

- **注意**：对于在计算时会合并N和T的模型（比如TSN、TSM），当`use_tensorrt=True`时，需要指定`batch_size`参数为batch_size*num_seg。

    ```bash
    python3.7 tools/predict.py --input_file data/example.avi \
                               --config configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml \
                               --model_file inference/ppTSM/ppTSM.pdmodel \
                               --params_file inference/ppTSM/ppTSM.pdiparams \
                               --batch_size 8 \
                               --use_gpu=True \
                               --use_tensorrt=True
    ```
## 参考论文

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), Geoffrey Hinton, Oriol Vinyals, Jeff Dean
