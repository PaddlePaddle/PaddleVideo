[简体中文](../../zh-CN/dataset/fsd.md) | English

# Figure Skating Dataset

- [Introduction](#Introduction)
- [Download](#Download)

---


## Introduction

In figure skating, compared with other sports, human posture and trajectory show the characteristics of strong complexity, which is helpful to the research of fine-grained action recognition tasks.

For FSD Dataset, all video materials are collected from the Figure Skating Championships from 2017 to 2018. The frame rate of the video is uniformly standardized to 30 frames per second, and the image size is 1080 * 720 to ensure the relative consistency of the dataset. After that, we use the 2D pose estimation algorithm Open Pose to extract frame by frame key points from the video, and finally save the data in `.npy` format.

The directory structure of training dataset and test dataset is as follows:

```txt
train_data.npy        # 2922
train_label.npy       # 2922
test_A_data.npy       # 628
test_B_data.npy       # 634
```

`train_label.npy` can be read using `np.load()`, each element is an integer variable with a value between 0-29, representing the label of the action. `data.npy` can be read using `np.load()`, return a tensor with the shape of `N×C×T×V×M`, the specific meaning of each dimension is as follows:

| Dimension | Size | Meaning	| Notes |
| :---- | :----: | :----: | :---- |
| N	| N	| Number of samples | - |
| C | 3	| The coordinates and confidence of each joint point respectively |	rescale to -1~1 |
| T	| 1500 |	 The duration of the action	| The actual length of some actions may be less than 1500, in such case we will pad 0 to ensure the unity of T dimension. |
| V |	25 | Number of joint points |	See the skeleton example below for the meaning of specific joint points. |
| M |	1	|  Number of athletes	| - |


skeleton example：

<div align="left">
  <img src="../../images/skeleton_example.png" width="180px"/><br>
</div>



## Download

You can get the download link after registering on the [competition homepage](https://www.datafountain.cn/competitions/519).

| Set | Data | Label	|
| :---- | :----: | :----: |
| Train	| [train_data.npy](https://videotag.bj.bcebos.com/Data/FSD_train_data.npy)	| [train_label.npy](https://videotag.bj.bcebos.com/Data/FSD_train_label.npy) |
| TestA	| comming soon	| comming soon |


> RGB datasets would not be provided for copyright reasons.
