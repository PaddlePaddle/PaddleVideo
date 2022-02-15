[简体中文](../../zh-CN/dataset/ActivityNet.md) | English

# ActivityNet data preparation

- [Dataset introduction](#Dataset introduction)
- [Data download and processing] (#Data download and processing)

## Dataset introduction

ActivityNet is a dataset for large-scale video understanding tasks, which can be used for tasks such as action localization, action recognition, etc.


## Data download and processing
- The BMN model uses the processed ActivityNet 1.3 dataset. There are two ways to use it:
    1. Using our processed ActivityNet 1.3 dataset (compressed package is about 5.5G), each video has corresponding action labels, duration intervals, duration frames, duration seconds and other information
        Download with the following command:
        ```bash
        wget https://paddlemodels.bj.bcebos.com/video_detection/bmn_feat.tar.gz # Download the processed video feature data
        wget https://paddlemodels.bj.bcebos.com/video_detection/activitynet_1.3_annotations.json # Download the processed label data
        ````
        Or click the following hyperlinks to download:

        [Video feature data](https://paddlemodels.bj.bcebos.com/video_detection/bmn_feat.tar.gz)
        [Video feature data](https://paddlemodels.bj.bcebos.com/video_detection/activitynet_1.3_annotations.json)

    2. Extract features by yourself

        First refer to [Download Instructions](https://github.com/activitynet/ActivityNet/tree/master/Crawler) to download the original dataset. When training this model, you need to use TSN to extract features from the source files first. You can [self-extract](https://github.com/yjxiong/temporal-segment-networks) video frame and optical flow information, and the pre-trained TSN model can be downloaded from [here](https://github.com/ yjxiong/anet2016-cuhk) download.

    In the end, `19228` video feature npy files should be obtained, corresponding to the `19228` label information in the `activitynet_1.3_annotations.json` file.

    The information in the `activitynet_1.3_annotations.json` tag file is as follows:
    ````json
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
            "duration_second": 61.7189999999999994,
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
    ````

- Create a new `data/bmn_data` folder, and then unzip the video feature data after downloading and put it in this folder, and finally it should be organized into the following form:
    ````
    PaddleVideo
    ├── data
    │   ├── bmn_data
    │   │   ├── fix_feat_100
    │   │   │   ├── v___c8enCfzqw.npy
    │   │   │   ├── v___dXUJsj3yo.npy
    │   │   │   ├── ...
    │   │   ├── activitynet_1.3_annotations.json
    ````

- Finally, modify the `feat_path` field in the configuration file configs/localization/bmn.yaml to specify the feature directory path, and the `file_path` field to specify the label file path.
