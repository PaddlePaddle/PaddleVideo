[简体中文](../../zh-CN/dataset/Oxford_RobotCar.md) | English

# Oxford-RobotCar-for-ADDS data preparation

- [Introduction](#Introduction)
- [Data Set Download](#Download)
- [Preprocessing](#Preprocessing)
- [1. Image De-distortion](#1-Image-de-distortion)
- [2. Dynamic frame filter](#2-Dynamic-frame-filter)
- [3. Image Rename](#3-Image-Rename)
- [4. Preparation for Day-Pseudo Night Image Pair](#4-Day-Pseudo-Night-Image-Pair-Preparation)


## Introduction

[Oxford RobotCar Dataset](https://robotcar-dataset.robots.ox.ac.uk/) is a large-scale autonomous driving data set that contains a large amount of data in different autonomous driving scenarios.

What is used here is to filter a part of the data used for day-night depth estimation from the original Oxford RobotCar data set, namely Oxford-RobotCar-for-ADDS.

If you want to use Oxford-RobotCar-for-ADDS, please cite the following papers:
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

## Download

1. Download the left eye image of Bumblebee XB3 in the sequence [2014-12-09](https://robotcar-dataset.robots.ox.ac.uk/datasets/2014-12-09-13-21-02/) as For the training set of the daytime scene, the downloaded images are decompressed in the same folder.
2. Download the left eye image of Bumblebee XB3 in the sequence [2014-12-16](https://robotcar-dataset.robots.ox.ac.uk/datasets/2014-12-16-18-44-24/) as The training set of the night scene, the downloaded images are unzipped in the same folder.
3. The images and depth truth values ​​of the validation set are filtered from the original data set and downloaded from the link we gave. (The data download links are below)
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
    the original raw data download links:
    ```shell
    # data in day
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

    # data in night
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

## Preprocessing

### 1-Image-de-distortion

Use the official toolbox [robotcar-dataset-sdk](https://github.com/ori-mrg/robotcar-dataset-sdk/tree/master/python) to pair the sequence 2014-12-09 and 2014-12- The image of 16 is de-distorted.


### 2-Dynamic-frame-filter

Since we use the self-supervised method, we need to filter out dynamic frames for training. The filtering principle is that the inter-frame pose change is greater than 0.1m and it is considered a dynamic frame. After filtering, the sequence of the training set is obtained.


### 3-Image-Rename

Rename the original image timestamp to a continuous number sequence. For daytime scene correspondence, see [1209_all_files.txt](https://videotag.bj.bcebos.com/Data/ADDS/1209_all_files.txt), for night scene correspondence, see [1216_all_files.txt](https://videotag.bj.bcebos.com/Data/ADDS/1216_all_files.txt). The renamed data format is as follows:
```
├── oxford_processing
    ├── day_train_all #Day training image folder (day_train_all.7z.001 ~ day_train_all.7z.012)
    ├── night_train_all #Night training image folder (night_train_all.7z.001 ~ day_train_all.7z.015)
    ├── day_val_451 #Daytime verification image folder (day_val_451.7z)
    ├── day_val_451_gt #Daytime verification depth truth value folder (day_val_451_gt.7z)
    ├── night_val_411 #night verification image folder (night_val_411.7z)
    └── night_val_411_gt #Night verification depth truth value folder (night_val_411_gt.7z)
```

annotation files download links are below:
```shell
https://videotag.bj.bcebos.com/Data/ADDS/train_files.txt
https://videotag.bj.bcebos.com/Data/ADDS/val_day_files.txt
https://videotag.bj.bcebos.com/Data/ADDS/val_night_files.txt
```

The sequence used for training and verification is as follows:

```
splits/oxford_day/train_files.txt # training sequence during the day
splits/oxford_night/train_files.txt # training sequence at night
splits/oxford_day_451/val_files.txt # verification sequence during the day
splits/oxford_night_411/val_files.txt # night verification sequence
```

### 4-Day-Pseudo-Night-Image-Pair-Preparation

In order to use our framework to extract the common information of day and night images, we use [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to generate day-pseudo-night image pairs, where pseudo-night The night images corresponding to the daytime generated for CycleGAN, all images are scaled to 192x640, the night images are enhanced with histogram equalization, 75 epochs are trained, and the Oxford-RobotCar-for-ADDS is finally obtained. The generated day-pseudo-night image pair The data format is as follows, which can be directly used for training and verification of ADDS-DepthNet:
```
├── oxford_processing_forADDS
    ├── day_train_all #Day training image folder (day_train_all.7z.001 ~ day_train_all.7z.002)
    ├── night_train_all #Night training image folder (night_train_all.7z.001 ~ day_train_all.7z.002)
    ├── day_val_451 #Daytime verification image folder (day_val_451.7z)
    ├── day_val_451_gt #Daytime verification depth truth value folder (day_val_451_gt.7z)
    ├── night_val_411 #night verification image folder (night_val_411.7z)
    └── night_val_411_gt #Night verification depth truth value folder (night_val_411_gt.7z)
data
└── oxford
    ├── splits
        ├── train_files.txt
        ├── val_day_files.txt
        └── val_night_files.txt
    └── oxford_processing_forADDS
        ├── day_train_all/      #Day training image folder (from day_train_all.7z.001 ~ day_train_all.7z.002)
        ├── night_train_all/    #Night training image folder (from night_train_all.7z.001 ~ day_train_all.7z.002)
        ├── day_val_451/        #Daytime verification image folder (from day_val_451.7z)
        ├── day_val_451_gt/     #Daytime verification depth truth value folder (from day_val_451_gt.7z)
        ├── night_val_411/      #night verification image folder (from night_val_411.7z)
        └── night_val_411_gt/   #Night verification depth truth value folder (from night_val_411_gt.7z)

```

The sequences used for training and verification are consistent with the foregoing.
