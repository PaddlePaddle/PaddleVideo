[English]() | 简体中文

# YouTube-8M数据准备

- [数据集简介](#数据集简介)
- [数据集下载](#数据集下载)
- [数据格式转化](#数据格式转化)


## 数据集简介

YouTube-8M 是一个大规模视频分类数据集，包含800多万个视频url，标签体系涵盖3800多种知识图谱实体，1个视频对应多个标签(平均3-4个)，使用机器进行标注。
视频特点:
- 每个视频的长度在120s到500s之间
由于视频数据量太大，因此预先使用图像分类模型提取了frame-level特征，用xxx提取了video-level特征，并使用PCA对特征进行了降维处理。音频特征？
> 这里用到的是YouTube-8M 2018年更新之后的数据集。  
  

## 数据集下载  
请使用Youtube-8M官方链接分别下载训练集（http://us.data.yt8m.org/2/frame/train/index.html ） 和  
验证集（http://us.data.yt8m.org/2/frame/validate/index.html ）。  
每个链接里各提供了3844个文件的下载地址，用户也可以使用官方提供的下载脚本下载数据。  
数据下载完成后，将会得到3844个训练数据文件和3844个验证数据文件（TFRecord格式）。   

## 数据格式转化
我们将下载的TFRecord文件转化为pickle文件以便PaddlePaddle使用。
为了加速，需要将TFRecord文件格式转成了pickle格式，请使用转化脚本：tf2pkl.py。  
然后将pkl拆分为单视频一个文件，请使用拆分脚本：split_yt8m.py。  
（ https://github.com/PaddlePaddle/PaddleVideo/blob/main/data/yt8m/split_yt8m.py ）
