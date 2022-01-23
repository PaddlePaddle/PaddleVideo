# MSR-VTT 数据准备

MSR-VTT 数据相关准备，包括MSR-VTT数据下载和数据下载后文件组织结构。

## 数据下载

MSR-VTT(Microsoft Research Video to Text) 是一个包含视频及字幕的大规模数据集，由来自20个类别的10,000个视频片段组成，每个视频片段由20个英文句子注释。我们使用9000个视频片段用于训练，1000个用于测试。更多详细信息可以参考网站：[MSRVTT](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)

为了方便使用，我们提供的数据版本已对MSR-VTT数据集中对视频进行了特征提取。

首先，请确保在 `data` 目录下，输入如下命令，下载数据集。

```bash
bash download_features.sh
```

下载完成后，data目录下文件组织形式如下：

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

## 参考论文
- Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid. Multi-modal transformer for video retrieval. In ECCV, 2020.
