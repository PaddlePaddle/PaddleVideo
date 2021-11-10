English | [简体中文](../../zh-CN/dataset/youtube8m.md)

# YouTube-8M Data Preparation

- [Introduction](#Introduction)
- [Download](#Download)
- [Conversion](#Conversion)


## Introduction

YouTube-8M is a large-scale video classification data set, containing more than 8 million video URLs. The tag system covers more than 3800 knowledge graph entities. One video corresponds to multiple tags (3-4 on average) and is labeled by machine.

**The length of each video is between 120s and 500s
Due to the large amount of video data, the image classification model was used to extract frame-level features in advance, and PCA was used to reduce the dimensionality of the features to obtain multi-frame 1024-dimensional features. Similarly, the audio model was used to obtain multi-frame 128-dimensional features. Audio characteristics. **
> The dataset used here is the updated YouTube-8M data set in 2018 (May 2018 version (current): 6.1M videos, 3862 classes, 3.0 labels/video, 2.6B audio-visual features).
  

## Download
1. Create a new directory for storing features (take the PaddleVideo directory as an example)
    ```bash
    cd data/yt8m
    mkdir frame
    cd frame
    ```
2. Download the training and validation set to the frame folder
    ```bash
    curl data.yt8m.org/download.py | partition=2/frame/train mirror=asia python
    curl data.yt8m.org/download.py | partition=2/frame/validate mirror=asia python
    ```
    The download process is shown in the figure
    ![image](https://user-images.githubusercontent.com/23737287/140709613-1e2d6ec0-a82e-474d-b220-7803065b0153.png)

    After the data download is complete, you will get 3844 training data files and 3844 verification data files (TFRecord format)

## Conversion
1. Install tensorflow to read tfrecord data
    ```bash
    python3.7 -m pip install tensorflow-gpu==1.14.0
    ```
2. Convert the downloaded TFRecord file into a pickle file for PaddlePaddle to use
    ```bash
    cd .. # From the frame directory back to the yt8m directory
    python3.7 tf2pkl.py ./frame ./pkl_frame/ # Convert train*.tfrecord and validate*.tfrecord in the frame folder to pkl format
    ```
3. Generate a single pkl file path set, and split pkl into multiple small pkl files based on this file, and generate the final split pkl file path required
    ```bash
    ls pkl_frame/train*.pkl> train.list # Write the path of train*.pkl to train.list
    ls pkl_frame/validate*.pkl> val.list # Write the path of validate*.pkl into val.list

    python3.7 split_yt8m.py train.list # Split each train*.pkl into multiple train*_split*.pkl
    python3.7 split_yt8m.py val.list # Split each validate*.pkl into multiple validate*_split*.pkl
    
    ls pkl_frame/train*_split*.pkl> train.list # Rewrite the path of train*_split*.pkl into train.list
    ls pkl_frame/validate*_split*.pkl> val.list # Rewrite the path of validate*_split*.pkl into val.list
    ``` 
