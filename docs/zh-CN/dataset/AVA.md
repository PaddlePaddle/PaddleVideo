[English](../../en/dataset/AVA.md) | 简体中文
# AVA数据准备
此文档主要介绍AVA数据集的相关准备流程。主要介绍 AVA数据集的视频文件下载，标注文件准备，视频文件切分
视频文件提取帧数据，以及拉取提名文件等。在开始之前，请把当前工作目录设定在 `$PaddleVideo/data/ava/shell`

---

## 1.  视频数据下载
想要获取更多有关AVA数据集的信息，您可以访问其官方网站[AVA](https://research.google.com/ava/index.html).
至于数据集下载，您可以参看考[AVA Download](https://github.com/cvdfoundation/ava-dataset) ，该Repo详细介绍了AVA视频数据的下载方法.
我们也提供了视频文件的下载脚本：

```shell
bash download_videos.sh
```

为了方便用户，我们将视频文件以zip包的形式上传到百度网盘，您可以直接进行下载 [Link]() <sup>coming soon</sup>.


**注意: 您自己下载的视频文件应当被放置在`data/ava/videos`文件夹下**  

---
## 2.准备标注文件

接下来，您可以使用下面的脚本来准备标注文件

```shell
bash download_annotations.sh
```

该脚本会默认下载`ava_v2.1.zip`，如果您想下载`v2.2`,您可以使用：

```shell
VERSION=2.2 bash download_annotations.sh
```

**注意：事实上，我们也同样在百度网盘中提供了该标注文件，所以您无需自己下载** 

---
## 3. 切分视频文件

以帧率30fps,切分视频文件从第15分钟到第30分钟

```shell
bash cut_videos.sh
```
---

## 4. 提取RGB帧

您可以通过以下的脚本使用`ffmpeg`来提取RGB帧.

```shell
bash extract_rgb_frames.sh
```

---

## 5.拉取提名文件

这个脚本来自于Facbook研究院[Long-Term Feature Banks](https://github.com/facebookresearch/video-long-term-feature-banks). 
您可以使用如下的脚本来获取预计算的提名文件列表。

```shell
bash fetch_ava_proposals.sh
```

---
## 6.目录结构

经过整个AVA数据处理流程后，您可以获得AVA的帧文件，视频文件和标注文件

整个项目(AVA)的目录结构如下所示：

```
PaddleVideo
├── configs
├── paddlevideo
├── docs
├── tools
├── data
│   ├── ava
│   │   ├── annotations
│   │   |   ├── ava_dense_proposals_train.FAIR.recall_93.9.pkl
│   │   |   ├── ava_dense_proposals_val.FAIR.recall_93.9.pkl
│   │   |   ├── ava_dense_proposals_test.FAIR.recall_93.9.pkl
│   │   |   ├── ava_train_v2.1.csv
│   │   |   ├── ava_val_v2.1.csv
│   │   |   ├── ava_train_excluded_timestamps_v2.1.csv
│   │   |   ├── ava_val_excluded_timestamps_v2.1.csv
│   │   |   ├── ava_action_list_v2.1_for_activitynet_2018.pbtxt
│   │   ├── videos
│   │   │   ├── 053oq2xB3oU.mkv
│   │   │   ├── 0f39OWEqJ24.mp4
│   │   │   ├── ...
│   │   ├── videos_15min
│   │   │   ├── 053oq2xB3oU.mkv
│   │   │   ├── 0f39OWEqJ24.mp4
│   │   │   ├── ...
│   │   ├── rawframes
│   │   │   ├── 053oq2xB3oU
|   │   │   │   ├── img_00001.jpg
|   │   │   │   ├── img_00002.jpg
|   │   │   │   ├── ...
```