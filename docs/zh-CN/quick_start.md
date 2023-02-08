简体中文 | [English](../en/quick_start.md)

# PaddleVideo快速开始

- [1. 安装](#1)
  - [1.1 安装PaddlePaddle](#11)
  - [1.2 安装PaddleVideo Whl包](#12)
- [2. 便捷使用](#2)
  - [2.1 命令行使用](#21)
  - [2.2 Python脚本使用](#22)
- [3.参数介绍](#3)
- [4.常见问题](#4)

## 1. 安装

<a name="11"></a>
### 1.1 安装PaddlePaddle

- 您的机器安装的是CUDA9或CUDA10，请运行以下命令安装

  ```bash
  python3.7 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
  ```

- 您的机器是CPU，请运行以下命令安装

  ```bash
  python3.7 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
  ```

更多的版本需求，请参照[飞桨官网安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

<a name="12"></a>
### 1.2 安装PaddleVideo whl包
- 方式1: 使用pypi安装（建议使用）

```bash
pip3.7 install ppvideo==2.3.0
```


- 方式2: 本地打包whl文件并安装
```bash
python3.7 setup.py bdist_wheel
python3.7 -m pip install dist/ppvideo-2.3.0-py3-none-any.whl
```


## 2. 便捷使用

<a name="21"></a>
### 2.1 命令行使用

安装完成后，运行如下脚本命令：
```bash
ppvideo --model_name='ppTSM_v2' --use_gpu=False --video_file='data/example.avi'
```

- 上述代码使用`PP-TSM_v2`模型，基于`CPU`，对`data/example.avi`示例文件进行预测。

- 示例视频长度约10s，抽帧策略采用分段抽帧，即先将视频按时间轴等分成16段，每段抽取一帧，所有帧组合之后，输入网络进行预测。

运行结果如下：

```
Current video file: data/example.avi
        top-1 classes: [5]
        top-1 scores: [1.]
        top-1 label names: ['archery']
```

可以看到，使用在Kinetics-400上训练好的`PP-TSM_v2`模型对`data/example.avi`进行行为识别，输出的top1类别id为`5`，置信度为`1.0`，预测类别名称为`archery`。

<a name="22"></a>
### 2.2 python脚本使用

安装完成后，运行如下示例代码：

```python
from ppvideo import PaddleVideo
clas = PaddleVideo(model_name='ppTSM_v2', use_gpu=False)
video_file='data/example.avi'
clas.predict(video_file)
```

上述代码使用`PP-TSM_v2`模型，基于`CPU`，对`data/example.avi`示例文件进行预测，运行结果如下：

```
Current video file: data/example.avi
        top-1 classes: [5]
        top-1 scores: [1.]
        top-1 label names: ['archery']
```

可以看到，使用在Kinetics-400上训练好的`PP-TSM_v2`模型对`data/example.avi`进行预测，输出的top1类别id为`5`，置信度为`1.0`，预测类别名称为`archery`。

<a name="3"></a>
## 3. 参数介绍

| 参数名称 | 参数类型 | 参数含义 |
| :---: | :---: | :--- |
| model_name | str | 可选，模型名称，`'ppTSM`'或`'ppTSM_v2'`。 如果不指定，需要通过`model_file`和`params_file`，提供自己的推理模型文件路径进行推理。 |
| video_file | str | 必选，视频文件路径，支持格式：单个视频文件路径，包含多个视频的文件夹。 |
| use_gpu | bool | 是否使用GPU，默认为True。 |
| num_seg | int | TSM分段采样策略中segment的数量，同时也是视频中抽帧的数量，8对应`ppTSM`模型，16对应`ppTSM_v2`模型，默认为16。 |
| short_size | int |  帧的短边尺寸大小，默认为256。|
| target_size | int | 帧的目标尺寸大小，默认为224。|
| model_file | str | 可选，推理模型的模型文件(`.pdmodel`)的路径。|
| params_file | str | 可选，推理模型的参数文件(`.pdiparams`)的路径。|
| batch_size | int | Batch size, 默认为1。|
| use_fp16 | bool | 是否使用float16，默认为False。|
| use_tensorrt | bool| 是否使用Tensorrt，默认为False。|
| gpu_mem | int | GPU使用显存大小，默认为8000。|
| enable_mkldnn | bool | 是否使用MKLDNN，默认为False。|
| top_k | int | 指定返回的top_k，默认为1。|
| label_name_path | str | 类别id和类别名称对应关系文件。默认使用Kinetics-400数据集使用的标签文件`data/k400/Kinetics-400_label_list.txt`，可参考以上格式替换成自己的标签文件。|

示例命令1：
```bash
ppvideo --model_name='ppTSM_v2' --num_seg=16 --video_file="data/mp4" --batch_size=2  --top_k=5
```
- 命令表示使用`PP-TSM_v2`模型，对`data/mp4`文件夹下的所有视频文件进行推理，`batch_size`为2，输出`top5`结果。
- `ppTSM`对应的`num_seg`为8，`ppTSM_v2`对应的`num_seg`为16。
- 使用GPU预测，占用显存约为`1400MB`。

输出示例：
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

示例命令2：
```bash
ppvideo --model_name='ppTSM' --num_seg=8 --video_file="data/mp4" --batch_size=2  --top_k=5
```
- 命令表示使用`ppTSM`模型进行推理。

<a name="4"></a>
## 4. 常见问题

1. 在下载opecv-python的过程中你可能遇到困难，可以尝试使用其他源进行安装：
```
python3.7 -m pip install opencv-python==4.2.0.32 -i https://pypi.doubanio.com/simple
```
