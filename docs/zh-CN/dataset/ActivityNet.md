简体中文 | [English](../../en/dataset/ActivityNet.md)

# BMN模型数据使用说明

BMN模型使用ActivityNet 1.3数据集，使用方法有如下两种方式：

方式一：

我们提供了处理好的视频特征和对应的标签文件，请下载特征数据[bmn\_feat](https://paddlemodels.bj.bcebos.com/video_detection/bmn_feat.tar.gz)和标签数据[label](https://paddlemodels.bj.bcebos.com/video_detection/activitynet_1.3_annotations.json)，并相应地修改配置文件configs/localization/bmn.yaml中的`feat_path`字段指定特征文件路径，通过`file_path`字段指定标签文件路径。

方式二：自行提取特征

首先参考[下载说明](https://github.com/activitynet/ActivityNet/tree/master/Crawler)下载原始数据集。在训练此模型时，需要先使用TSN对源文件抽取特征。可以[自行抽取](https://github.com/yjxiong/temporal-segment-networks)视频帧及光流信息，预训练好的TSN模型可从[此处](https://github.com/yjxiong/anet2016-cuhk)下载。

