[English](../../../en/model_zoo/recognition/stgcn.md)  | 简体中文

# ST-GCN基于骨骼的行为识别模型

---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)
- [参考论文](#参考论文)


## 模型简介

ST-GCN是AAAI 2018提出的经典的基于骨骼的行为识别模型，通过将图卷积应用在具有拓扑结构的人体骨骼数据上，使用时空图卷积提取时空特征进行行为识别，极大地提升了基于骨骼的行为识别任务精度。

我们提供了详尽理论及代码讲解，并可使用免费在线GPU算力资源，一键运行的AI Studio Notebook项目， 使用链接：[基于飞桨实现花样滑冰选手骨骼点动作识别大赛baseline](https://aistudio.baidu.com/aistudio/projectdetail/2417717?contributionType=1)

<div align="center">
<img src="../../../images/st-gcn.png" height=200 width=950 hspace='10'/> <br />
</div>


## 数据准备

花样滑冰比赛数据下载及准备请参考[花样滑冰数据准备](../../dataset/fsd.md)

NTU-RGBD数据下载及准备请参考[NTU-RGBD数据准备](../../dataset/ntu-rgbd.md)


## 模型训练

### 花样滑冰数据集训练

- 花样滑冰数据集使用单卡训练，启动命令如下:

```bash
python3.7 main.py -c configs/recognition/stgcn/stgcn_fsd.yaml
```

- 由于赛事未提供验证集数据，因此训练时不做valid。

- 您可以自定义修改参数配置，以达到在不同的数据集上进行训练/测试的目的，参数用法请参考[config](../../tutorials/config.md)。


### NTU-RGBD数据集训练

- NTU-RGBD数据集使用4卡训练，启动命令如下:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_stgcn  main.py  --validate -c configs/recognition/stgcn/stgcn_ntucs.yaml
```

- 配置文件`stgcn_ntucs.yaml`为NTU-RGB+D数据集按cross-subject划分方式对应的训练配置。


## 模型测试
### 花样滑冰数据集模型测试

- 模型测试的启动命令如下：

```bash
python3.7 main.py --test -c configs/recognition/stgcn/stgcn_fsd.yaml -w output/STGCN/STGCN_epoch_00090.pdparams
```

- 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。

- 评估结果保存在submission.csv文件中，可在[评测官网](https://aistudio.baidu.com/aistudio/competition/detail/115)提交查看得分。

模型在花样滑冰数据集上baseline实验精度如下:

Test_Data| Top-1 | checkpoints |
| :----: | :----: | :---- |
| Test_A | 59.07 | [STGCN_fsd.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/STGCN_fsd.pdparams) |


### NTU-RGB+D数据集模型测试

- 模型测试的启动命令如下：

```bash
python3.7 main.py --test -c configs/recognition/stgcn/stgcn_ntucs.yaml -w output/STGCN/STGCN_best.pdparams
```

- 通过`-c`参数指定配置文件，通过`-w`指定权重存放路径进行模型测试。

模型在NTU-RGB+D数据集上实验精度如下:

| split | Top-1 | checkpoints |
| :----: | :----: | :---- |
| cross-subject | 82.28 | [STGCN_ntucs.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/STGCN_ntucs.pdparams) |

### 视频数据测试

- 提供从视频中提取骨骼点数据的方法，方便用户自行提取数据进行测试。

花样滑冰数据提取采用了openpose，通过其提供的demo或是相应的api来实现数据的提取，因此需要用户配置openpose环境。
如下是通过花样滑冰数据集构建项目[Skeleton Scripts](https://github.com/HaxiSnake/skeleton_scripts)提取骨骼点数据方法的具体介绍。

#### step1 安装openpose

- 参考：https://github.com/CMU-Perceptual-Computing-Lab/openpose  

#### step2 测试openpose提供demo

- 这里通过测试openpose的demo程序来验证是否安装成功。

demo1：检测视频中身体骨骼点（以linux系统为例）：

```bash
./build/examples/openpose/openpose.bin --video examples_video.avi --write_json output/ --display 0 --render_pose 0
```

执行成功之后会在output/路径下生成视频每一帧骨骼点数据的json文件。

demo2：检测视频中身体+面部+手部骨骼点（以linux系统为例）：

```bash
./build/examples/openpose/openpose.bin --video examples_video.avi --write_json output/ --display 0 --render_pose 0 --face --hand
```

执行成功之后会在output/路径下生成视频每一帧身体+面部+手部骨骼点数据的json文件。

#### step3 视频及相关信息处理

- 由于[Skeleton Scripts](https://github.com/HaxiSnake/skeleton_scripts)为制作花样滑冰数据集所用，因此此处步骤可能存在不同程度误差，实际请用户自行调试代码。

将要转化的花样滑冰视频储存到[Skeleton Scripts](https://github.com/HaxiSnake/skeleton_scripts)的指定路径（可自行创建）：
```bash
./skating2.0/skating63/
```

同时需要用户自行完成对视频信息的提取，保存为label_skating63.csv文件，储存到如下路径中（可自行创建）：

```bash
./skating2.0/skating63/
./skating2.0/skating63_openpose_result/
```

label_skating63.csv中格式如下：

| 动作分类 | 视频文件名 | 视频帧数 | 动作标签 |
| :----: | :----: | :----: | :---- |

此处用户只需要输入视频文件名（无需后缀，默认后缀名为.mp4，其他格式需自行更改代码)，其他三项定义为空字符串即可，不同表项之间通过 ',' 分割。

#### step4 执行skating_convert.py:

- 注意，这一步需要根据用户对openpose的配置进行代码的更改，主要修改项为openpose路径、openpose-demo路径等，具体详见代码。

本脚步原理是调用openpose提供的demo提取视频中的骨骼点，并进行数据格式清洗，最后将每个视频的提取结果结果打包成json文件，json文件储存在如下路径：

```bash
./skating2.0/skating63_openpose_result/label_skating63_data/
```

#### step5 执行skating_gendata.py:

将json文件整理为npy文件并保存，多个视频文件将保存为一个npy文件，保存路径为：

```bash
./skating2.0/skating63_openpose_result/skeleton_file/
```

- 通过上述步骤就可以将视频数据转化为无标签的骨骼点数据。

- 最后用户只需将npy数据输入送入网络开始模型测试，亦可通过预测引擎推理（如下）。

## 模型推理

### 导出inference模型

```bash
python3.7 tools/export_model.py -c configs/recognition/stgcn/stgcn_fsd.yaml \
                                -p data/STGCN_fsd.pdparams \
                                -o inference/STGCN
```

上述命令将生成预测所需的模型结构文件`STGCN.pdmodel`和模型权重文件`STGCN.pdiparams`。

- 各参数含义可参考[模型推理方法](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### 使用预测引擎推理

```bash
python3.7 tools/predict.py --input_file data/fsd10/example_skeleton.npy \
                           --config configs/recognition/stgcn/stgcn_fsd.yaml \
                           --model_file inference/STGCN/STGCN.pdmodel \
                           --params_file inference/STGCN/STGCN.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

输出示例如下:

```
Current video file: data/fsd10/example_skeleton.npy
        top-1 class: 27
        top-1 score: 0.9912770986557007
```

可以看到，使用在花样滑冰数据集上训练好的ST-GCN模型对`data/example_skeleton.npy`进行预测，输出的top1类别id为`27`，置信度为0.9912770986557007。


## 参考论文

- [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455), Sijie Yan, Yuanjun Xiong, Dahua Lin
