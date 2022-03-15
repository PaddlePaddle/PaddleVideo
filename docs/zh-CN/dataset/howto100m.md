# HowTo100M 数据准备

HowTo100M 数据相关准备，包括HowTo100M数据下载和数据下载后文件组织结构。

## 数据下载

HowTo100M 从1.2M Youtube 教学视频中切分出136M包含字幕的视频片段，涵盖23k活动类型，包括做饭、手工制作、日常护理、园艺、健身等等，数据集约10T大小。

因为完整数据集体积过大，这里我们只提供少量数据，供大家跑通训练前向。如需下载全量数据，请参考：[HowTo100M](https://www.di.ens.fr/willow/research/howto100m/)

为了方便使用，我们提供的数据版本已对HowTo100M数据集中的物体特征和动作特征进行了特征提取。 

首先，请确保在 `data/howto100m` 目录下，输入如下命令，下载数据集。

```bash
bash download_features.sh
```

下载完成后，data目录下文件组织形式如下：

```
├── data
|   ├── howto100m
|   │   ├── actbert_train_data.npy
|   │   ├── caption_train.json
|   |   ├── caption_val.json

```

## 参考论文
- Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In ICCV, 2019.
