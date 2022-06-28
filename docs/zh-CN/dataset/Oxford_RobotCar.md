[English](../../en/dataset/Oxford_RobotCar.md) | 简体中文

# Oxford-RobotCar-for-ADDS数据准备

- [数据集简介](#数据集简介)
- [数据集下载](#数据集下载)
- [数据预处理](#数据预处理)
- [1. 图像去畸变](#1-图像去畸变)
- [2. 动态帧筛选](#2-动态帧筛选)
- [3. 图像重命名](#3-图像重命名)
- [4. 白天-伪夜晚图像对准备](#4-白天-伪夜晚图像对准备)


## 数据集简介

[Oxford RobotCar Dataset](https://robotcar-dataset.robots.ox.ac.uk/) 是一个大规模自动驾驶数据集, 包含了大量不同自动驾驶场景下的数据.

这里用到的是从原始的Oxford RobotCar数据集中筛选出一部分用于白天-夜晚深度估计的数据, 即Oxford-RobotCar-for-ADDS.

如果您要使用Oxford-RobotCar-for-ADDS, 请引用以下论文:
```latex
@article{maddern20171,
  title={1 year, 1000 km: The oxford robotcar dataset},
  author={Maddern, Will and Pascoe, Geoffrey and Linegar, Chris and Newman, Paul},
  journal={The International Journal of Robotics Research},
  volume={36},
  number={1},
  pages={3--15},
  year={2017},
  publisher={SAGE Publications Sage UK: London, England}
}
```
```latex
@inproceedings{liu2021self,
  title={Self-supervised Monocular Depth Estimation for All Day Images using Domain Separation},
  author={Liu, Lina and Song, Xibin and Wang, Mengmeng and Liu, Yong and Zhang, Liangjun},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12737--12746},
  year={2021}
}
```

## 数据集下载

1. 下载序列[2014-12-09](https://robotcar-dataset.robots.ox.ac.uk/datasets/2014-12-09-13-21-02/) 中Bumblebee XB3的左目图像作为白天场景的训练集, 下载好的图像解压在同一文件夹下.
2. 下载序列[2014-12-16](https://robotcar-dataset.robots.ox.ac.uk/datasets/2014-12-16-18-44-24/) 中Bumblebee XB3的左目图像作为夜晚场景的训练集, 下载好的图像解压在同一文件夹下.
3. 验证集的图像和深度真值从原始数据集中筛选, 下载地址如下：
    ```shell
    https://videotag.bj.bcebos.com/Data/ADDS/1209_all_files.txt
    https://videotag.bj.bcebos.com/Data/ADDS/1216_all_files.txt
    https://videotag.bj.bcebos.com/Data/ADDS/day_train_all.7z.001
    https://videotag.bj.bcebos.com/Data/ADDS/day_train_all.7z.002
    https://videotag.bj.bcebos.com/Data/ADDS/day_train_all_fake_night.7z.001
    https://videotag.bj.bcebos.com/Data/ADDS/day_train_all_fake_night.7z.002
    https://videotag.bj.bcebos.com/Data/ADDS/day_val_451.7z
    https://videotag.bj.bcebos.com/Data/ADDS/day_val_451_gt.7z
    https://videotag.bj.bcebos.com/Data/ADDS/night_val_411.7z
    https://videotag.bj.bcebos.com/Data/ADDS/night_val_411_gt.7z
    ```
    附原始未处理数据下载地址：
    ```shell
    # 白天数据
    https://videotag.bj.bcebos.com/Data/original-ADDS/day_train_all.7z.001
    https://videotag.bj.bcebos.com/Data/original-ADDS/day_train_all.7z.002
    https://videotag.bj.bcebos.com/Data/original-ADDS/day_train_all.7z.003
    https://videotag.bj.bcebos.com/Data/original-ADDS/day_train_all.7z.004
    https://videotag.bj.bcebos.com/Data/original-ADDS/day_train_all.7z.005
    https://videotag.bj.bcebos.com/Data/original-ADDS/day_train_all.7z.006
    https://videotag.bj.bcebos.com/Data/original-ADDS/day_train_all.7z.007
    https://videotag.bj.bcebos.com/Data/original-ADDS/day_train_all.7z.008
    https://videotag.bj.bcebos.com/Data/original-ADDS/day_train_all.7z.009
    https://videotag.bj.bcebos.com/Data/original-ADDS/day_train_all.7z.010
    https://videotag.bj.bcebos.com/Data/original-ADDS/day_train_all.7z.011
    https://videotag.bj.bcebos.com/Data/original-ADDS/day_train_all.7z.012

    # 夜晚数据
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.001
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.002
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.003
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.004
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.005
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.006
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.007
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.008
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.009
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.010
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.011
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.012
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.013
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.014
    https://videotag.bj.bcebos.com/Data/original-ADDS/night_train_all.7z.015
    ```
## 数据预处理

#### 1. 图像去畸变

使用官方提供的工具箱[robotcar-dataset-sdk](https://github.com/ori-mrg/robotcar-dataset-sdk/tree/master/python) 对序列2014-12-09和2014-12-16的图像完成去畸变.


#### 2. 动态帧筛选

由于我们使用自监督的方法, 需要筛选出动态帧用于训练. 筛选原则为帧间位姿变化大于0.1m则认为是动态帧. 经过筛选后获得训练集的序列.


#### 3. 图像重命名

将原始图像时间戳重命名为连续数字序列. 白天场景对应关系见[1209_all_files.txt](https://videotag.bj.bcebos.com/Data/ADDS/1209_all_files.txt), 夜晚场景对应关系见[1216_all_files.txt](https://videotag.bj.bcebos.com/Data/ADDS/1216_all_files.txt). 重命名后的数据格式如下:
```
├── oxford_processing
    ├── day_train_all      #白天训练图像文件夹 (day_train_all.7z.001 ~ day_train_all.7z.012)
    ├── night_train_all    #夜晚训练图像文件夹 (night_train_all.7z.001 ~ day_train_all.7z.015)
    ├── day_val_451        #白天验证图像文件夹 (day_val_451.7z)
    ├── day_val_451_gt     #白天验证深度真值文件夹 (day_val_451_gt.7z)
    ├── night_val_411      #夜晚验证图像文件夹 (night_val_411.7z)
    └── night_val_411_gt   #夜晚验证深度真值文件夹 (night_val_411_gt.7z)
```

其中用于训练和验证的序列如下:

```
splits/oxford_day/train_files.txt       # 白天训练序列
splits/oxford_night/train_files.txt     # 夜晚训练序列
splits/oxford_day_451/val_files.txt     # 白天验证序列
splits/oxford_night_411/val_files.txt   # 夜晚验证序列
```
训练所用路径文本的下载地址：
```shell
https://videotag.bj.bcebos.com/Data/ADDS/train_files.txt
https://videotag.bj.bcebos.com/Data/ADDS/val_day_files.txt
https://videotag.bj.bcebos.com/Data/ADDS/val_night_files.txt
```

#### 4. 白天-伪夜晚图像对准备

为了用我们的框架提取出白天和夜晚图像的共有信息,我们用[CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)生成白天-伪夜晚图像对,其中伪夜晚为CycleGAN生成的与白天对应的夜晚图像, 所有图像都缩放为192x640, 夜晚图像用直方图均衡化增强, 训练75个epoch, 最终得到Oxford-RobotCar-for-ADDS. 生成的白天-伪夜晚图像对数据格式如下,可直接用于ADDS-DepthNet的训练和验证:
```
data
└── oxford
    ├── splits
        ├── train_files.txt
        ├── val_day_files.txt
        └── val_night_files.txt
    └── oxford_processing_forADDS
        ├── day_train_all/      #白天训练图像文件夹 (解压自day_train_all.7z.001 ~ day_train_all.7z.002)
        ├── night_train_all/    #夜晚训练图像文件夹 (解压自night_train_all.7z.001 ~ day_train_all.7z.002)
        ├── day_val_451/        #白天验证图像文件夹 (解压自day_val_451.7z)
        ├── day_val_451_gt/     #白天验证深度真值文件夹 (解压自day_val_451_gt.7z)
        ├── night_val_411/      #夜晚验证图像文件夹 (解压自night_val_411.7z)
        └── night_val_411_gt/   #夜晚验证深度真值文件夹 (解压自night_val_411_gt.7z)
```

其中用于训练和验证的序列与前述保持一致.
