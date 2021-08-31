[简体中文](../../zh-CN/dataset/ntu-rgbd.md) | English

# NTU-RGB+D Preparation

- [Introduction](#Introduction)
- [Download](#Download)

---


## Introduction

NTU-RGB+D contains 60 action classes and 56,880 video samples for skeleton-based action recognition. Please refer to its official website[NTU-RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) for more details.

The dataset contains two splits when dividing the training set and test set. For Cross-subject, the dataset is divided according to character id, with 40320 samples in training set and 16560 samples in test set. For Cross-view, the dataset is divided according to camera division. The samples collected by cameras 2 and 3 are training sets, including 37930 samples, and the samples collected by camera 1 are test sets, including 18960 samples.

## Download

We provide the download link of the processed dataset [NTU-RGB-D.tar](https://videotag.bj.bcebos.com/Data/NTU-RGB-D.tar)(~3.1G). Please download and unzip, the directory structure is as follows：

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

> This is a copies from [st-gcn](https://github.com/open-mmlab/mmskeleton/blob/master/doc/SKELETON_DATA.md).
