[简体中文](../../zh-CN/dataset/msrvtt.md) | English

# MSR-VTT Preparation

- [Introduction](#1.1)
- [Download for T2VLAD](#1.2)
- [Download for ActBERT](#1.3)
- [Reference](#1.4)


<a name="1.1"></a>
## Introduction

MSR-VTT(Microsoft Research Video to Text) is a large-scale dataset containing videos and subtitles, which is composed of 10000 video clips from 20 categories, and each video clip is annotated with 20 English sentences. We used 9000 video clips for training and 1000 for testing. For more details, please refer to the website: [MSRVTT](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)

<a name="1.2"></a>
## Download for T2VLAD

[T2VLAD doc](../../../applications/T2VLAD/README_en.md)

For ease of use, we provided extracted features of video.

First, make sure to enter the following command in the `applications/T2VLAD/data` directory to download the dataset.

```bash
bash download_features.sh
```

After downloading, the files in the data directory are organized as follows:

```
├── data
|   ├── MSR-VTT
|   │   ├── raw-captions.pkl
|   │   ├── train_list_jsfusion.txt
|   │   ├── val_list_jsfusion.txt
|   │   ├── aggregated_text_feats
|   |   |   ├── w2v_MSRVTT_openAIGPT.pickle
|   |   ├── mmt_feats
|   │   │   ├── features.audio.pkl
|   │   │   ├── features.face_agg.pkl
|   │   │   ├── features.flos_agg.pkl
|   │   │   ├── features.ocr.pkl
|   │   │   ├── features.rgb_agg.pkl
|   │   │   ├── features.s3d.pkl
|   │   │   ├── features.scene.pkl
|   │   │   ├── features.speech.pkl

```
<a name="1.3"></a>
## Download for ActBERT

[ActBERT doc](../model_zoo/multimodal/actbert.md)

Download data features:
```
wget https://videotag.bj.bcebos.com/Data/ActBERT/msrvtt_test.lmdb.tar
wget https://videotag.bj.bcebos.com/Data/ActBERT/MSRVTT_JSFUSION_test.csv
```

Decompress the `msrvtt_test.lmdb.tar`：
```
tar -zxvf msrvtt_test.lmdb.tar
```

The files in the data directory are organized as follows:

```
├── data
|   ├── MSR-VTT
|   │   ├── MSRVTT_JSFUSION_test.csv
|   │   ├── msrvtt_test.lmdb
|   │       ├── data.mdb
|   │       ├── lock.mdb
```

<a name="1.4"></a>
## Reference
- Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid. Multi-modal transformer for video retrieval. In ECCV, 2020.
