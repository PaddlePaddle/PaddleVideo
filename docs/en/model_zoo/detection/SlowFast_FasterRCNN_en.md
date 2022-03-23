[简体中文](../../../zh-CN/model_zoo/detection/SlowFast_FasterRCNN.md) | English

# SlowFast_FasterRCNN

## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)

Before getting started, you need to install additional dependencies as follows:
```bash
python -m pip install moviepy
python -m pip install et_xmlfile
python -m pip install paddledet
```

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

### 3 Download Proposals

```
bash  fetch_ava_proposals.sh
```

### 4 Cut Videos

```
bash  cut_videos.sh
```

### 5 Extract Frames

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
* `-w`: weights of model. The pretrained model can be downloaded from the table below;
* `--validate`: evaluate model during training.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=logdir.ava main.py --validate -w paddle.init_param.pdparams -c configs/detection/ava/ava.yaml
```

## Test

Test model based on the best model:
```
python main.py --test \
   -w output/AVA_SlowFast_FastRcnn/AVA_SlowFast_FastRcnn_best.pdparams \
   -c configs/detection/ava/ava.yaml
```


| architecture | depth | Pretrain Model |  frame length x sample rate  | MAP | AVA version | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |
| SlowFast | R50 | [Kinetics 400](https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast_8*8.pdparams) | 8 x 8 | 23.2 | 2.1 | [`link`](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/SlowFastRCNN_AVA.pdparams) |


## Inference

The action detection of this project is divided into two stages. In the first stage, humans' proposals are obtained, and then input into the SlowFast+FasterRCNN model for action recognition.

For human detection，you can use the trained model in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection).

Install PaddleDetection:
```
cd PaddleDetection/
pip install -r requirements.txt
!python setup.py install
```

Download detection model:
```
# faster_rcnn_r50_fpn_1x_coco as an example
wget https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams
```

export model:
```
python tools/export_model.py \
  -c configs/detection/ava/ava.yaml \
  -o inference_output \
  -p output/AVA_SlowFast_FastRcnn/AVA_SlowFast_FastRcnn_best.pdparams
```

inference based on the exported model:
```
python tools/predict.py \
    -c configs/detection/ava/ava.yaml \
    --input_file "data/-IELREHXDEMO.mp4" \
    --model_file "inference_output/AVA_SlowFast_FastRcnn.pdmodel" \
    --params_file "inference_output/AVA_SlowFast_FastRcnn.pdiparams" \
    --use_gpu=True \
    --use_tensorrt=False
```
