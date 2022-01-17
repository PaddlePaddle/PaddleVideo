# Paddle-Anti-UAV
Anti-UAV base on PaddleDetection

## Background
UAVs are very popular and we can see them in many public spaces, such as parks and playgrounds. Most people use UAVs for taking photos.
However, many areas like airport forbiden UAVs since they are potentially dangerous. In this case, we need to detect the flying UAVs in
these areas.

In this repository, we show how to train a detection model using [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection).

## Data preparation
The dataset can be found [here](https://anti-uav.github.io/dataset/). We direcly download the ```test-dev``` split composed of 140 videos
train the detection model.
* Download the ```test-dev``` dataset.
* Run `unzip Anti_UAV_test_dev.zip -d Anti_UAV`.
* Run `python get_image_label.py`. In this step, you may change the path to the videos and the value of `interval`.

After the above steps, you will get a MSCOCO-style datasst for object detection.

## Install PaddleDetection
Please refer to this [link](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/docs/tutorials/INSTALL.md).

We use `python=3.7`, `Paddle=2.2.1`, `CUDA=10.2`.

## Train PP-YOLO
We use [PP-YOLO](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo) as the detector.
* Run `git clone https://github.com/PaddlePaddle/PaddleDetection.git`. Note that you should finish this step when you install PaddleDetection.
* Move the anti-UAV dataset to `dataset`.
* Move `anti_uav.yml` to `configs/datasets`, move `ppyolo_r50vd_dcn_1x_antiuav.yml` to `configs/ppyolo` and move `ppyolo_r50vd_dcn_antiuav.yml`
to `configs/ppyolo/_base`.
* Keep the value of `anchors` in `configs/ppyolo/_base/ppyolo_reader.yml` the same as `ppyolo_r50vd_dcn_antiuav.yml`.
* Run `python -m paddle.distributed.launch --log_dir=./ppyolo_dygraph/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_antiuav.yml &>ppyolo_dygraph.log 2>&1 &`.
Note that you may change the arguments, such as `batch_size` and `gups`.

## Inference
Please refer to the infernce section on this [webpage](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/docs/tutorials/GETTING_STARTED.md). You can just switch the configeration file and trained model to your own files.

![](https://github.com/qingzwang/Paddle-Anti-UAV/blob/main/demo1.gif)
![](https://github.com/qingzwang/Paddle-Anti-UAV/blob/main/demo.gif)
