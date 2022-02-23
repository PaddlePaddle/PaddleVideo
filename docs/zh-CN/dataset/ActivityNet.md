[English](../../en/dataset/ActivityNet.md) | 简体中文

# ActivityNet数据准备

- [数据集介绍](#数据集介绍)
- [数据下载与处理](#数据下载与处理)

## 数据集介绍

ActivityNet是一个用于大规模视频理解任务的数据集，可用于动作定位、动作识别等任务。


## 数据下载与处理
1. BMN模型使用的是处理过后的ActivityNet 1.3数据集，有如下两种使用方法：
    - 使用我们处理好的ActivityNet 1.3数据集(压缩包约5.5G)，每一个视频有对应的动作标签、持续区间、持续帧数、持续秒数等信息
        使用以下命令下载：
        ```bash
        wget https://paddlemodels.bj.bcebos.com/video_detection/bmn_feat.tar.gz  # 下载处理好的视频特征数据
        wget https://paddlemodels.bj.bcebos.com/video_detection/activitynet_1.3_annotations.json  # 下载处理好的标签数据
        ```
        或者点击以下超链接下载：

        [视频特征数据](https://paddlemodels.bj.bcebos.com/video_detection/bmn_feat.tar.gz)
        [视频特征数据](https://paddlemodels.bj.bcebos.com/video_detection/activitynet_1.3_annotations.json)

        然后解压下下好的视频特征压缩包
        ```bash
        tar -xf bmn_feat.tar.gz
        ```

    - 自行提取特征

        首先参考[下载说明](https://github.com/activitynet/ActivityNet/tree/master/Crawler)下载原始数据集。在训练此模型时，需要先使用TSN对源文件抽取特征。可以[自行抽取](https://github.com/yjxiong/temporal-segment-networks)视频帧及光流信息，预训练好的TSN模型可从[此处](https://github.com/yjxiong/anet2016-cuhk)下载。


    `activitynet_1.3_annotations.json`标签文件内的信息如下所示：
    ```json
    {
        "v_QOlSCBRmfWY": {
            "duration_second": 82.73,
            "subset": "training",
            "duration_frame": 2067,
            "annotations": [{
                "segment": [6.195294851794072, 77.73085420904837],
                "label": "Ballet"
            }],
            "feature_frame": 2064
        },
        "v_ehGHCYKzyZ8": {
            "duration_second": 61.718999999999994,
            "subset": "training",
            "duration_frame": 1822,
            "annotations": [{
                "segment": [43.95990729267573, 45.401932082395355],
                "label": "Doing crunches"
            }],
            "feature_frame": 1808
        },
        ...,
        ...
    }
    ```
    最终应该能得到`19228`个视频特征npy文件，对应`activitynet_1.3_annotations.json`文件中的`19228`个标签信息。

2. 新建`data/bmn_data`文件夹，再将下载完毕后将视频特征数据解压出来放入该文件夹下，最终应该组织成以下形式：
    ```
    PaddleVideo
    ├── data
    │   ├── bmn_data
    │   │   ├── fix_feat_100
    │   │   │   ├── v___c8enCfzqw.npy
    │   │   │   ├── v___dXUJsj3yo.npy
    │   │   │   ├── ...
    │   │   │
    │   │   └── activitynet_1.3_annotations.json
    ```

3. 最后修改配置文件configs/localization/bmn.yaml中的`feat_path`字段指定特征文件夹路径，通过`file_path`字段指定标签文件路径。


