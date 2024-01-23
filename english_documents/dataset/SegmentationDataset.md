English | [简体中文](../../zh-CN/dataset/SegmentationDataset.md)

# Video Action Segmentation Dataset

The video motion segmentation model uses breakfast, 50salads and gtea data sets. The use method is to use the features extracted by the pre training model, which can be obtained from the ms-tcn official code base.[feat](https://zenodo.org/record/3625992#.Xiv9jGhKhPY)

- Dataset tree
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

- data tree
```txt
─── data
    ├── 50salads
    ├── breakfast
    ├── gtea
    └── ...
```
