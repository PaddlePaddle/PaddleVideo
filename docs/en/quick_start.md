English | [简体中文](../zh-CN/quick_start.md)

# PaddleVide Quick Start

- [1. Installation](#1)
  - [1.1 Install PaddlePaddle](#11)
  - [1.2 Install PaddleVideo Whl Package](#12)
- [2. Easy-to-Use](#2)
  - [2.1 Use by Command Line](#21)
  - [2.2 Use by Python Code](#22)
- [3. Arguments description](#3)
- [4.QA](#4)

## 1. Installation

<a name="11"></a>
### 1.1 Install PaddlePaddle

- If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install

  ```bash
  python3.7 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
  ```

- If you have no available GPU on your machine, please run the following command to install the CPU version

  ```bash
  python3.7 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
  ```

For more software version requirements, please refer to the instructions in [Installation Document](https://www.paddlepaddle.org.cn/install/quick) for operation.


<a name="12"></a>
### 1.2 Install PaddleVideo Whl Package
- option1: use pypi（recommand）

```bash
python3.7 -m pip install paddlevideo==1.0.0
```


- option2: build and install locally
```bash
python3.7 setup.py bdist_wheel
python3.7 -m pip install dist/paddlevideo-1.0.0-py3-none-any.whl
```


## 2. Easy-to-Use

<a name="21"></a>
### 2.1 Use by Command Line

Run shell command：
```bash
ppvideo --model_name='ppTSM_v2' --use_gpu=False --video_file='data/example.avi'
```

- This command use `PP-TSM_v2` model to infer `data/example.avi` file in `CPU`.

- The length of the example video is about 10s. When inference, the video is first divided into 16 segments according to the time axis, then extract one frame from each segment. Finally all frames are combined and feeded into the network.

Results：

```
Current video file: data/example.avi
        top-1 classes: [5]
        top-1 scores: [1.]
        top-1 label names: ['archery']
```

As you can see, use `PP-TSM_v2` trained on Kinetics-400 to predict `data/example.avi` video，top1 prediction class_id is `5`, scores is `1.0`, class name is `archery`.

<a name="22"></a>
### 2.2 Use by Python Code

Run python code：

```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_name='ppTSM_v2', use_gpu=False)
video_file='data/example.avi'
clas.predict(video_file)
```

- This code use `PP-TSM_v2` model to infer `data/example.avi` file in `CPU`.

Results:
```
Current video file: data/example.avi
        top-1 classes: [5]
        top-1 scores: [1.]
        top-1 label names: ['archery']
```

As you can see, use `PP-TSM_v2` trained on Kinetics-400 to predict `data/example.avi` video，top1 prediction class_id is `5`, scores is `1.0`, class name is `archery`.

<a name="3"></a>
## 3. Arguments description

| name | type | description |
| :---: | :---: | :--- |
| model_name | str | optional, model name, `'ppTSM'` or `'ppTSM_v2'`. If None, please specify the path of your inference model by args `model_file` and `params_file`. |
| video_file | str | required, Video file path, supported format: single video file path, or folder containing multiple videos. |
| use_gpu | bool | whether to use GPU，default True。 |
| num_seg | int | The number of segments used in the TSM model, which is also the number of frames extracted from the video. 8 for `ppTSM`, 16 for `ppTSM_v2`, default 16. |
| short_size | int |  short size of frame, default 256.|
| target_size | int | target size of frame, default 224.|
| model_file | str | optional，inference model(`.pdmodel`)path. |
| params_file | str | optional, inference modle(`.pdiparams`) path. |
| batch_size | int | Batch size, default 1.|
| use_fp16 | bool | whether to use float16，default False.|
| use_tensorrt | bool| whether to use Tensorrt, default False.|
| gpu_mem | int | use GPU memory, default 8000.|
| enable_mkldnn | bool | whether to use MKLDNN, default False.|
| top_k | int | top_k, default 1. |
| label_name_path | str | This file consists the relation of class_id and class_name. Default use `data/k400/Kinetics-400_label_list.txt` of Kinetics-400. You can replace it with your own label file. |

command example1：
```bash
ppvideo --model_name='ppTSM_v2' --num_seg=16 --video_file="data/mp4" --batch_size=2  --top_k=5
```


Results：
```txt
Current video file: data/mp4/example3.avi
        top-5 classes: [  5 345 311 159 327]
        top-5 scores: [1.0000000e+00 1.0152016e-11 8.2871061e-14 6.7713670e-14 5.0752070e-14]
        top-5 label names: ['archery', 'sword_fighting', 'skipping_rope', 'hula_hooping', 'spray_painting']
Current video file: data/mp4/example2.avi
        top-5 classes: [  5 345 311 159 327]
        top-5 scores: [1.0000000e+00 1.0152016e-11 8.2871061e-14 6.7713670e-14 5.0752070e-14]
        top-5 label names: ['archery', 'sword_fighting', 'skipping_rope', 'hula_hooping', 'spray_painting']
Current video file: data/mp4/example.avi
        top-5 classes: [  5 345 311 159 327]
        top-5 scores: [1.0000000e+00 1.0152016e-11 8.2871061e-14 6.7713670e-14 5.0752070e-14]
        top-5 label names: ['archery', 'sword_fighting', 'skipping_rope', 'hula_hooping', 'spray_painting']
Current video file: data/mp4/example1.avi
        top-5 classes: [  5 345 311 159 327]
        top-5 scores: [1.0000000e+00 1.0152016e-11 8.2871061e-14 6.7713670e-14 5.0752070e-14]
        top-5 label names: ['archery', 'sword_fighting', 'skipping_rope', 'hula_hooping', 'spray_painting']
```

command example1：
```bash
ppvideo --model_name='ppTSM' --num_seg=8 --video_file="data/mp4" --batch_size=2  --top_k=5
```

<a name="4"></a>
## 4. QA

1. opecv-python Installation maybe slow, you can try:
```
python3.7 -m pip install opencv-python==4.2.0.32 -i https://pypi.doubanio.com/simple
```
