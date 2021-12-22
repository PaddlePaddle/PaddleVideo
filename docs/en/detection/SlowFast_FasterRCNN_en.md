[简体中文](../../../zh-CN/model_zoo/detection/SlowFast_FastRCNN.md) | English

# SlowFast_FasterRCNN

## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)


## Introduction

The [SlowFast](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/slowfast.md) model is one of the high-precision models in the video field. For action detection task, it is also neccessary to detect the person in current frame. Therefore, the SlowFast_FasterRCNN model takes human detection results and video frames as input, extracts spatiotemporal features through the SlowFast model, and then uses FasterRCNN's head gets the actions and positions of humans in the frame. 

The corresponding AI Studio Notebook Link：[基于SlowFast+FasterRCNN的动作识别](https://aistudio.baidu.com/aistudio/projectdetail/3267637?contributionType=1)

For details, please refer to the paper [SlowFast Networks for Video Recognition](https://arxiv.org/pdf/1812.03982.pdf).

## Data

We use [AVA dataset](https://research.google.com/ava/download.html) for action detection. The AVA v2.2 dataset contains 430 videos split into 235 for training, 64 for validation, and 131 for test. Each video has 15 minutes annotated in 1 second intervals. 

### 1 Dowload Videos
```
bash  download_videos.sh
```

### 2 Download Annotations
```
bash  download_annotations.sh
```

### 3 Extrac Frames
```
bash  extract_rgb_frames.sh
```

For AVA v2.1, there is a simple introduction to some key files：
* 'ava_videos_15min_frames' dir stores video frames extracted with FPS as the frame rate；
* 'ava_train_v2.1.csv' file stores the trainning annotations；
* 'ava_train_excluded_timestamps_v2.1.csv' file stores excluded timestamps；
* 'ava_dense_proposals_train.FAIR.recall_93.9.pkl' file stores humans' bboxes and scores of key frames；
* 'ava_action_list_v2.1_for_activitynet_2018.pbtxt' file stores为 action list.

## Train

* `-c`: config file path;
* `-w`: weights of model;
* `--validate`: evaluate model during training.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logdir.ava main.py --validate -w paddle.init_param.pdparams -c configs/detection/ava/ava.yaml
```

## Evaluate

Evaluate model based on the best model:
```
python main.py --valid \
   -w output/AVA_SlowFast_FastRcnn/AVA_SlowFast_FastRcnn_best.pdparams \
   -c configs/detection/ava/ava.yaml
```

## Test

The action detection of this project is divided into two stages. In the first stage, humans' proposals are obtained, and then input into the SlowFast+FasterRCNN model for action recognition.

For human detection，you can use the trained model in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection).

Before object detection, extract frames from video. The following code shows the realization of extracting one frame per secong:

```
import os
import os.path as osp
import cv2

timeRate = 1  # 1 frame per second 

def frame_extraction(video_path,target_dir):
    """Extract frames given video_path.
    Args:
        video_path (str): The video_path.
    """

    # if dir not exists, create target dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, '{:05d}.jpg')
    vid = cv2.VideoCapture(video_path)

    FPS = vid.get(5)
    print("FPS",FPS)

    frameRate = int(FPS) * timeRate #每隔多少帧保存一个，采样率
    
    frames = []
    frame_paths = []

    flag, frame = vid.read()
    cnt = 0
    index = 1
    while flag:
        if cnt%frameRate == 0: #sample
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

The persons in the extracted video frame can be obtained through the trained model provided by PaddleDetection.

The SlowFast_FasterRCNN model needs to input densely sampled video frame data, and extract the dense video frames through the following command:

1. First params: the dir of extracted frames；
1. Second params: the path of video;
1. Third params: FPS.

```
bash extract_video_frames.sh './data/frames_30fps/1j20qq1JyX4' \
 './data/1j20qq1JyX4.mp4' 30
```

Action detection based on the result of FPS frame extraction and the detection result:
- detection_result_dir: the dir of detection result;
- frame_dir: the dir of extracted frames. 

```
python tools/infer.py \
  -c configs/detection/ava/ava.yaml \
  -w ./output/AVA_SlowFast_FastRcnn/AVA_SlowFast_FastRcnn_best.pdparams \
  --detection_result_dir ./data/detection_result/1j20qq1JyX4 \
  --frame_dir ./data/frames_30fps/1j20qq1JyX4
```

