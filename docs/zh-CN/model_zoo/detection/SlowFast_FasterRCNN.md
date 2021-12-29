简体中文 | [English](../../../en/model_zoo/detection/SlowFast_FastRCNN_en.md)

# SlowFast_FasterRCNN

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型测试)
- [模型测试](#模型推理)


## 模型简介

[SlowFast](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/slowfast.md)模型是视频领域的高精度模型之一，对于动作识别任务，还需要检测出当前画面人物，因此SlowFast_FasterRCNN模型以人的检测结果和视频数据为输入，通过SlowFast模型提取时空特征，然后利用FasterRCNN的head得到画面中每个人的动作和位置。

我们提供了详尽理论及代码讲解，并可使用免费在线GPU算力资源，一键运行的AI Studio Notebook项目，使用链接：[基于SlowFast+FasterRCNN的动作识别](https://aistudio.baidu.com/aistudio/projectdetail/3267637?contributionType=1)

详细内容请参考论文[SlowFast Networks for Video Recognition](https://arxiv.org/pdf/1812.03982.pdf)中AVA Action Detection相关内容。

## 数据准备

本项目利用[AVA数据集](https://research.google.com/ava/download.html)进行动作检测。AVA v2.2数据集包括430个视频，其中235个用于训练，64个用于验证，131个用于测试。对每个视频中15分钟的帧进行了标注，每秒标注一帧。标注文件格式为CSV。

### 1 下载视频
```
bash  download_videos.sh
```

### 2 下载标注
```
bash  download_annotations.sh
```

### 3 提取视频帧
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

* `-c`后面的参数是配置文件的路径。
* `-w`后面的参数是finetuning或者测试时的权重。
* `--validate`参数表示在训练过程中进行模型评估。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logdir.ava main.py --validate -w paddle.init_param.pdparams -c configs/detection/ava/ava.yaml
```

## 模型测试

基于训练好的模型进行测试：
```
python main.py --valid \
   -w output/AVA_SlowFast_FastRcnn/AVA_SlowFast_FastRcnn_best.pdparams \
   -c configs/detection/ava/ava.yaml
```

## 模型推理

本项目动作识别分成两个阶段，第一个阶段得到人的proposals，然后再输入到SlowFast+FasterRCNN模型中进行动作识别。

对于画面中人的检测，可利用[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)中的模型。

在进行目标检测前，先将视频抽帧，下面代码展示了每秒抽一帧实现：

```
import os
import os.path as osp
import cv2

timeRate = 1  # 截取视频帧的时间间隔（这里是每隔1秒截取一帧）

def frame_extraction(video_path,target_dir):
    """Extract frames given video_path.
    Args:
        video_path (str): The video_path.
    """

    # 保存帧的目录不存在，创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, '{:05d}.jpg')
    vid = cv2.VideoCapture(video_path)

    FPS = vid.get(5)
    print("视频帧率：",FPS)

    frameRate = int(FPS) * timeRate #每隔多少帧保存一个，采样率

    frames = []
    frame_paths = []

    flag, frame = vid.read()
    cnt = 0
    index = 1
    while flag:
        if cnt%frameRate == 0: #间隔采样
           frames.append(frame)
           frame_path = frame_tmpl.format(index)
           frame_paths.append(frame_path)
           cv2.imwrite(frame_path, frame)
           index+=1
        cnt += 1
        flag, frame = vid.read()
    return frame_paths, frames

video_path = './data/1j20qq1JyX4.mp4'
target_dir = './data/tmp/1j20qq1JyX4'
frame_paths, frames = frame_extraction(video_path,target_dir)
print("抽帧总数：",len(frames))
```

通过PaddleDetection提供的训练好的模型即可得到抽取的视频帧中的目标。

SlowFast_FasterRCNN模型需要输入密集采样的视频帧数据，通过下面的命令提取视频密集帧：

1. 第一个参数是抽帧结果存放路径；
1. 第二个参数是视频路径；
1. 第三个参数是FPS。

```
bash extract_video_frames.sh './data/frames_30fps/1j20qq1JyX4' \
 './data/1j20qq1JyX4.mp4' 30
```

基于以FPS抽帧的结果和检测结果进行动作识别：
- detection_result_dir 检测结果存放路径；
- frame_dir 抽帧结果存放路径。

```
python tools/infer.py \
  -c configs/detection/ava/ava.yaml \
  -w ./output/AVA_SlowFast_FastRcnn/AVA_SlowFast_FastRcnn_best.pdparams \
  --detection_result_dir ./data/detection_result/1j20qq1JyX4 \
  --frame_dir ./data/frames_30fps/1j20qq1JyX4
```
