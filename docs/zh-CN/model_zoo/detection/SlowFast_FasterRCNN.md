简体中文 | [English](../../../en/model_zoo/detection/SlowFast_FasterRCNN_en.md)

# SlowFast_FasterRCNN

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型推理](#模型推理)

在开始使用之前，您需要按照以下命令安装额外的依赖包：
```bash
python -m pip install moviepy
python -m pip install et_xmlfile
python -m pip install paddledet
```

## 模型简介

[SlowFast](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/slowfast.md)模型是视频领域的高精度模型之一，对于动作识别任务，还需要检测出当前画面人物，因此SlowFast_FasterRCNN模型以人的检测结果和视频数据为输入，通过SlowFast模型提取时空特征，然后利用FasterRCNN的head得到画面中每个人的动作和位置。

我们提供了详尽理论及代码讲解，并可使用免费在线GPU算力资源，一键运行的AI Studio Notebook项目，使用链接：[基于SlowFast+FasterRCNN的动作识别](https://aistudio.baidu.com/aistudio/projectdetail/3267637?contributionType=1)

详细内容请参考论文[SlowFast Networks for Video Recognition](https://arxiv.org/pdf/1812.03982.pdf)中AVA Action Detection相关内容。

## 数据准备

本项目利用[AVA数据集](https://research.google.com/ava/download.html)进行动作检测。AVA v2.2数据集包括430个视频，其中235个用于训练，64个用于验证，131个用于测试。对每个视频中15分钟的帧进行了标注，每秒标注一帧。标注文件格式为CSV。

相关处理脚本在`data/ava/script`目录下。

### 1 下载视频
```
bash  download_videos.sh
```

### 2 下载标注
```
bash  download_annotations.sh
```

### 3 下载检测结果

```
bash  fetch_ava_proposals.sh
```

### 4 视频切割
把下载的视频中第15分钟起后面的15分钟的片段切割出来：

```
bash  cut_videos.sh
```

### 5 提取视频帧
```
bash  extract_rgb_frames.sh
```

此处以AVA v2.1版本为例，进行关键文件介绍：
* ava_videos_15min_frames文件夹中存放以FPS为帧率抽取的视频帧；
* ava_train_v2.1.csv文件存放训练数据标注；
* ava_train_excluded_timestamps_v2.1.csv文件中存放废弃的时间戳数据；
* ava_dense_proposals_train.FAIR.recall_93.9.pkl文件中为每个关键帧中人的位置和置信度数据；
* ava_action_list_v2.1_for_activitynet_2018.pbtxt为动作类别数据。

## 模型训练

下载预训练模型：
```
wget https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast_8*8.pdparams
```


* `-c`后面的参数是配置文件的路径。
* `-w`后面的参数是finetuning或者测试时的权重，本案例将在Kinetics 400上训练的SlowFast R50模型作为预训练权重，通过下面的表格可获取。
* `--validate`参数表示在训练过程中进行模型评估。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logdir.ava main.py --validate -w SlowFast_8*8.pdparams -c configs/detection/ava/ava.yaml
```

## 模型评估

基于训练好的模型进行评估：
```
python main.py --test \
   -w output/AVA_SlowFast_FastRcnn/AVA_SlowFast_FastRcnn_best.pdparams \
   -c configs/detection/ava/ava.yaml
```

| architecture | depth | Pretrain Model |  frame length x sample rate  | MAP | AVA version | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |
| SlowFast | R50 | [Kinetics 400](https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast_8*8.pdparams) | 8 x 8 | 23.2 | 2.1 | [`link`](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/SlowFastRCNN_AVA.pdparams) |


## 模型推理

本项目动作识别分成两个阶段，第一个阶段得到人的proposals，然后再输入到SlowFast+FasterRCNN模型中进行动作识别。

对于画面中人的检测，可利用[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)中的模型。

PaddleDetection安装：
```
# 安装其他依赖
cd PaddleDetection/
pip install -r requirements.txt

# 编译安装paddledet
python setup.py install
```

下载训练好的检测模型参数：
```
wget https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams
```

导出模型：

```
!python tools/export_model.py \
  -c configs/detection/ava/ava.yaml \
  -o inference_output \
  -p output/AVA_SlowFast_FastRcnn/AVA_SlowFast_FastRcnn_best.pdparams
```

基于导出的模型做推理：

```
python tools/predict.py \
    -c configs/detection/ava/ava.yaml \
    --input_file "data/-IELREHXDEMO.mp4" \
    --model_file "inference_output/AVA_SlowFast_FastRcnn.pdmodel" \
    --params_file "inference_output/AVA_SlowFast_FastRcnn.pdiparams" \
    --use_gpu=True \
    --use_tensorrt=False
```
