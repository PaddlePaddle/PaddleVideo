[English](../../en/dataset/youtube8m.md) | 简体中文

# YouTube-8M数据准备

- [数据集简介](#数据集简介)
- [数据集下载](#数据集下载)
- [数据格式转化](#数据格式转化)


## 数据集简介

YouTube-8M 是一个大规模视频分类数据集，包含800多万个视频url，标签体系涵盖3800多种知识图谱实体，1个视频对应多个标签(平均3-4个)，使用机器进行标注。

**每个视频的长度在120s到500s之间
由于视频数据量太大，因此预先使用图像分类模型提取了frame-level的特征，并使用PCA对特征进行了降维处理得到多帧1024维的特征，类似地用音频模型处理得到多帧128维的音频特征。**
> 这里用到的是YouTube-8M 2018年更新之后的数据集（May 2018 version (current): 6.1M videos, 3862 classes, 3.0 labels/video, 2.6B audio-visual features）。  
  

## 数据集下载

1. 新建存放特征的目录（以PaddleVideo目录下为例）
    ```bash
    cd data/yt8m
    mkdir frame
    cd frame
    ```
2. 下载训练、验证集到frame文件夹中
    ```bash
    curl data.yt8m.org/download.py | partition=2/frame/train mirror=asia python
    curl data.yt8m.org/download.py | partition=2/frame/validate mirror=asia python
    ```
    下载过程如图所示
    ![image](https://user-images.githubusercontent.com/23737287/140709613-1e2d6ec0-a82e-474d-b220-7803065b0153.png)

    数据下载完成后，将会得到3844个训练数据文件和3844个验证数据文件（TFRecord格式）


## 数据格式转化
1. 安装tensorflow-gpu用于读入tfrecord数据
    ```bash
    python3.7 -m pip install tensorflow-gpu==1.14.0
    ```
3. 将下载的TFRecord文件转化为pickle文件以便PaddlePaddle使用
    ```bash
    cd .. # 从frame目录回到yt8m目录
    python3.7 tf2pkl.py ./frame ./pkl_frame/ # 将frame文件夹下的train*.tfrecord和validate*.tfrecord转化为pkl格式
    ```
2. 生成单个pkl文件路径集合，并根据此文件将pkl拆分为多个小pkl文件，并生成最终需要的拆分pkl文件路径
    ```bash
    ls pkl_frame/train*.pkl > train.list # 将train*.pkl的路径写入train.list
    ls pkl_frame/validate*.pkl > val.list # 将validate*.pkl的路径写入val.list

    python3.7 split_yt8m.py train.list # 拆分每个train*.pkl变成多个train*_split*.pkl
    python3.7 split_yt8m.py val.list # 拆分每个validate*.pkl变成多个validate*_split*.pkl
    
    ls pkl_frame/train*_split*.pkl > train.list # 将train*_split*.pkl的路径重新写入train.list
    ls pkl_frame/validate*_split*.pkl > val.list # 将validate*_split*.pkl的路径重新写入val.list
    ```

