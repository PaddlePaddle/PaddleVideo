[English](../../en/dataset/ntu-rgbd.md) | 简体中文

# NTU-RGB+D 数据准备

- [数据集介绍](#数据集介绍)
- [数据集下载](#数据集下载)

---


## 数据集介绍

NTU-RGB+D是基于骨骼的行为识别数据集，包含60个种类的动作，56880个样本，详细介绍可以参考其官方网站[NTU-RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/)。该数据集在划分训练集和测试集时采用了两种不同的划分标准。Cross-Subject按照人物ID划分，训练集40320个样本，测试集16560个样本。Cross-View安装相机划分，相机2和3采集的样本为训练集，包含37930个样本，相机1采集的样本为测试集，包含18960个样本。

## 数据集下载

我们提供处理好的数据集下载地址[NTU-RGB-D.tar](https://videotag.bj.bcebos.com/Data/NTU-RGB-D.tar)(~3.1G)，下载后解压，数据目录如下：

```txt
─── NTU-RGB-D
    ├── xsub
    │   ├── train_data.npy
    │   ├── train_label.pkl
    │   ├── val_data.npy
    │   └── val_label.pkl
    └── xview
        ├── train_data.npy
        ├── train_label.pkl
        ├── val_data.npy
        └── val_label.pkl
```

> 数据来源于[st-gcn](https://github.com/open-mmlab/mmskeleton/blob/master/doc/SKELETON_DATA.md)。
