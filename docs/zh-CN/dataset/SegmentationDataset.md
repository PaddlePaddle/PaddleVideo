简体中文 | [English](../../en/dataset/SegmentationDataset.md)

# 视频动作分割模型数据使用说明

视频动作分割模型使用breakfast、50salads和gtea数据集，使用方法为使用预训练模型提取的特征，可以从MS-TCN官方代码库中获取。[feat](https://zenodo.org/record/3625992#.Xiv9jGhKhPY)

- 数据集文件树形式
```txt
─── gtea
    ├── features
    │   ├── S1_Cheese_C1.npy
    │   ├── S1_Coffee_C1.npy
    │   ├── S1_CofHoney_C1.npy
    │   └── ...
    ├── groundTruth
    │   ├── S1_Cheese_C1.txt
    │   ├── S1_Coffee_C1.txt
    │   ├── S1_CofHoney_C1.txt
    │   └── ...
    ├── splits
    │   ├── test.split1.bundle
    │   ├── test.split2.bundle
    │   ├── test.split3.bundle
    │   └── ...
    └── mapping.txt
```

- 数据集存放文件树形式
```txt
─── data
    ├── 50salads
    ├── breakfast
    ├── gtea
    └── ...
```
