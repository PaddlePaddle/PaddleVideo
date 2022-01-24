[English](../../en/dataset/DAVIS2017.md) | 简体中文

# DAVIS2017 数据集准备

## 1.数据下载

下载 [DAVIS2017](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) 和 [scribbles](https://data.vision.ee.ethz.ch/csergi/share/DAVIS-Interactive/DAVIS-2017-scribbles-trainval.zip)到同一个文件夹中。请参阅[DAVIS](https://davischallenge.org/davis2017/code.html).

如果您需要文件"DAVIS2017/ImageSets/2017/v_a_l_instances.txt"，请参阅[google](https://drive.google.com/file/d/1aLPaQ_5lyAi3Lk3d2fOc_xewSrfcrQlc/view?usp=sharing)链接

## 2.目录结构

整个项目(Ma-Net)的目录结构如下所示：

```shell
PaddleVideo
├── configs
├── paddlevideo
├── docs
├── tools
├── data
│ 	└── DAVIS2017
│   │ 	├── Annotations
│   │ 	├── ImageSets
│   │ 	├── JPEGImages
│   │ 	└── Scribbles
```
