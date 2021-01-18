简体中文 | [English](../en/whl_en.md)
# paddlevideo包使用教程

## 快速开始

### 安装

使用pypi安装
```bash
pip install paddlevideo==0.0.1
```
**注意:** 在下载opecv-python的过程中你可能遇到困难，你可以尝试使用其他源进行安装，试一试：
```
pip install opencv-python==4.2.0.32 -i https://pypi.doubanio.com/simple
```

本地打包whl文件并安装
```bash
python3 setup.py bdist_wheel
pip3 install dist/paddlevideo-x.x.x-py3-none-any.whl
```

### 1. 快速开始

* 指定 `video_file='data/example.mp4'`, 使用飞桨提供的推理模型 `model_name='ppTSM'`


```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_name='ppTSM',use_gpu=False,use_tensorrt=False)
video_file='data/example.mp4.'
result=clas.predict(video_file)
print(result)
```

```
    >>> result
    [{'videoname': 'data/example.mp4', 'class_ids': [5], 'scores': [0.999963], 'label_names': ['archery']}]
```

* 使用命令行方式启动程序
```bash
ppvideo --model_name='ppTSM' --video_file='data/example.mp4'
```

```
    >>> result
    **********data/example.mp4**********
    [{'videoname': 'data/example.mp4', 'class_ids': [5], 'scores': [0.999963], 'label_names': ['archery']}]
```

### 2. 参数介绍
* model_name(str): 模型的名字. 如果不指定`model_file`和`params_file`你需要指定这个参数来使用飞桨提供的在K400数据集上预训练的模型，默认设置为ppTSM
* video_file(str): 视频文件路径. 支持：本地单一视频文件，包含多个视频文件的文件夹,numpy数组。
* use_gpu(bool): 是否使用GPU，默认为不使用。
* num_seg(int): TSN提出的分段采样策略中分段的数量。
* seg_len(int): 每个分段上采样的帧数。
* short_size(int): 将帧的短边调整为多少像素，默认为256。
* target_size(int): 调整帧的尺寸为目标尺寸，默认为224。
* normalize(bool): 是否对帧进行归一化。默认为True。
* model_file(str): 推理模型的模型文件(inference.pdmodel)的路径,如果不指定这个参数，你需要指定`model_name`来进行下载。
* params_file(str): 推理模型的参数文件(inference.pdiparams)的路径，如果不指定这个参数，你需要指定`model_name`来进行下载。
* batch_size(int): Batch size, 默认为1。
* use_fp16(bool): 是否使用float16，默认为False。
* use_tensorrt(bool): 是否使用Tensorrt，默认为False。
* gpu_mem(int): GPU使用显存大小，默认为8000。
* top_k(int): 指定返回的top_k,默认为1。
* enable_mkldnn(bool): 是否使用MKLDNN，默认为False。


### 3. 不同使用方式介绍

**我们提供两种不同的使用方式：1.使用python交互式编程 2.使用命令行方式**

* 查看帮助信息
```bash
ppvideo -h
```

* 使用用户指定的模型，你需要指定模型文件的路径 `model_file` 和参数文件路径 `params_file`

###### python
```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_file='user-specified model path',
    params_file='parmas path', use_gpu=False, use_tensorrt=False)
video_file = ''
result=clas.predict(video_file)
print(result)
```

###### bash
```bash
ppvideo --model_file='user-specified model path' --params_file='parmas path' --video_file='video path'
```

* 使用飞桨提供的推理模型进行预测，你需要通过指定 `model_name`参数来选择一个模型对ppvideo进行初始化，这时你不需要指定 `model_file`文件，你所选择的model预训练模型会自动下载到 `BASE_INFERENCE_MODEL_DIR`目录中以 `model_name`命名的文件夹下

###### python
```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_name='ppTSM',use_gpu=False, use_tensorrt=False)
video_file = ''
result=clas.predict(video_file)
print(result)
```

###### bash
```bash
ppvideo --model_name='ppTSM' --video_file='video path'
```

* 你可以将 `np.ndarray`形式的数组作为输入，同样以 `--video_file=np.ndarray`方式指定即可

###### python
```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_name='ppTSM',use_gpu=False, use_tensorrt=False)
video_file =np.ndarray
result=clas.predict(video_file)
```

###### bash
```bash
ppvideo --model_name='ppTSM' --video_file=np.ndarray
```

* 你可以将 `video_file`指定为一个包含多个视频文件的路径，同样也可以指定 `top_k`参数

###### python
```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_name='ppTSM',use_gpu=False, use_tensorrt=False,top_k=5)
video_file = '' # it can be video_file folder path which contains all of videos you want to predict.
result=clas.predict(video_file)
print(result)
```

###### bash
```bash
paddleclas --model_name='ppTSM' --video_file='video path' --top_k=5
```

* 你可以指定 `--label_name_path`为你自己的标签文件，**注意** 格式必须为(类别ID 类别名)

```
0 abseiling
1 air_drumming
2 answering_questions
3 applauding
4 applying_cream
5 archery
......
```

* 如果你使用的是飞桨提供的推理模型，你不需要指定`label_name_path`,程序将默认使用`data/k400/Kinetics-400_label_list.txt`；如果你想使用你自己训练的模型，你需要提供你自己的label文件，否则模型只能输出预测的分数而没有类别名称

###### python
```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_file= './inference.pdmodel',params_file = './inference.pdiparams',label_name_path='./data/k400/Kinetics-400_label_list.txt',use_gpu=False)
video_file = '' # it can be video_file folder path which contains all of videos you want to predict.
result=clas.predict(video_file)
print(result)
```
###### bash
```bash
ppvideo --model_file= './inference.pdmodel' --params_file = './inference.pdiparams' --video_file='video path' --label_name_path='./data/k400/Kinetics-400_label_list.txt'
```
###### python
```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_name='ppTSM',use_gpu=False)
video_file = '' # it can be video_file folder path which contains all of videos you want to predict.
result=clas.predict(video_file)
print(result)
```
###### bash
```bash
ppvideo --model_name='ppTSM' --video_file='video path'
```
