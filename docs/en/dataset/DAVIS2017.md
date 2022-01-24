[简体中文](../../zh-CN/dataset/DAVIS2017.md) | English

# DAVIS2017 Data Preparation

## 1.Data Download

Download [DAVIS2017](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) and [scribbles](https://data.vision.ee.ethz.ch/csergi/share/DAVIS-Interactive/DAVIS-2017-scribbles-trainval.zip) into one folder. Please refer to [DAVIS](https://davischallenge.org/davis2017/code.html).

If you need the file "DAVIS2017/ImageSets/2017/v_a_l_instances.txt", please refer to the link [google]( https://drive.google.com/file/d/1aLPaQ_5lyAi3Lk3d2fOc_xewSrfcrQlc/view?usp=sharing)

## 2.Folder Structure

In the context of the whole project (for Ma-Net only), the folder structure will look like:

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
