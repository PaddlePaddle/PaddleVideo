English | [简体中文](../../zh-CN/dataset/youtube8m.md)

# YouTube-8MData Preparation

- [Introduction](#Introduction)
- [Download](#Download)
- [Conversion](#Conversion)


## Introduction

YouTube-8M is a large-scale video classification data set, containing more than 8 million video URLs. The tag system covers more than 3800 knowledge graph entities. One video corresponds to multiple tags (3-4 on average) and is labeled by machine.
Video features:
-The length of each video is between 120s and 500s
Because the amount of video data is too large, the frame-level feature is extracted using the image classification model, the video-level feature is extracted with xxx, and the feature is reduced in dimensionality using PCA. Audio characteristics?
> The dataset used here is the updated YouTube-8M data set in 2018.
  

## Download
Please use the official Youtube-8M link to download the training set (http://us.data.yt8m.org/2/frame/train/index.html) and
Validation set (http://us.data.yt8m.org/2/frame/validate/index.html).
Each link provides download addresses of 3844 files, and users can also download data using the official download script.
After the data download is complete, you will get 3844 training data files and 3844 verification data files (TFRecord format).

## Conversion
We convert the downloaded TFRecord file into a pickle file for PaddlePaddle to use.
In order to speed up, you need to convert the TFRecord file format to pickle format, please use the conversion script: tf2pkl.py.
Then split pkl into a single video and a file, please use the split script: split_yt8m.py.
(Https://github.com/PaddlePaddle/PaddleVideo/blob/main/data/yt8m/split_yt8m.py)
