[简体中文](../../zh-CN/dataset/ntu-rgbd.md) | English

# NTU-RGB+D Preparation

- [Introduction](#Introduction)
- [ST-GCN Data Prepare](#ST-GCN_Data_Prepare)
- [CTR-GTCN Data Prepare](#CTR-GCN_Data_Prepare)

---


## Introduction

NTU-RGB+D contains 60 action classes and 56,880 video samples for skeleton-based action recognition. Please refer to its official website[NTU-RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) for more details.

The dataset contains two splits when dividing the training set and test set. For Cross-subject, the dataset is divided according to character id, with 40320 samples in training set and 16560 samples in test set. For Cross-view, the dataset is divided according to camera division. The samples collected by cameras 2 and 3 are training sets, including 37930 samples, and the samples collected by camera 1 are test sets, including 18960 samples.

## ST-GCN_Data_Prepare

ST-GCN data prepare preceduce are introducted follow.

### Download
We provide the download link of the processed dataset [NTU-RGB-D.tar](https://videotag.bj.bcebos.com/Data/NTU-RGB-D.tar)(~3.1G). Please download and unzip with ```tar -zxvf NTU-RGB-D.tar ``` , the directory structure is as follows：

```txt
─── NTU-RGB-D
    ├── xsub
    │   ├── train_data.npy
    │   ├── train_label.pkl
    │   ├── val_data.npy
    │   └── val_label.pkl
    └── xview
        ├── train_data.npy
        ├── train_label.pkl
        ├── val_data.npy
        └── val_label.pkl
```

> This is a copies from [st-gcn](https://github.com/open-mmlab/mmskeleton/blob/master/doc/SKELETON_DATA.md).

## CTR-GCN_Data_Prepare

CTR-GCN data prepare preceduce are introducted follow.

### Download

There is script `download_dataset.sh` to download the dataset from official website [NTU-RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) in dictory `data\ntu-rgb-d`.

```bash
sh data/ntu-rgb-d/download_dataset.sh
```

File tree:
```txt
─── ntu-rgb-d
    ├── download_dataset.sh
    ├── nturgb+d_skeletons
    │   ├── S001C001P001R001A001.skeleton
    │   ├── S001C001P001R001A002.skeleton
    │   ├── S001C001P001R001A003.skeleton
    │   ├── S001C001P001R001A004.skeleton
    │   ├── S001C001P001R001A005.skeleton
    │   ├── S001C001P001R001A006.skeleton
    │   ├── S001C001P001R001A007.skeleton
    │   ├── ....
    │   └── S017C003P020R002A060.skeleton
    ├── get_raw_denoised_data.py
    ├── get_raw_skes_data.py
    ├── seq_transformation.py
    └── statistics
        ├── camera.txt
        ├── label.txt
        ├── performer.txt
        ├── replication.txt
        ├── setup.txt
        └── skes_available_name.txt

```

### Prepare

run follow script, then data will be precessed to the data format need by CTR-GCN.

> Note：if make dataset by yourself, please prepare `data/ntu-rgb-d/statistics/skes_available_name.txt`, which is the list of skeletons files that will be precessed.

```bash
cd ./data/ntu-rgb-d
# Get skeleton of each performer
python get_raw_skes_data.py
# Remove the bad skeleton
python get_raw_denoised_data.py
# Transform the skeleton to the center of the first frame
python seq_transformation.py
```

File tree:

```txt
─── ntu-rgb-d
    ├── download_dataset.sh
    ├── nturgb+d_skeletons
    │   ├── S001C001P001R001A001.skeleton
    │   ├── S001C001P001R001A002.skeleton
    │   ├── S001C001P001R001A003.skeleton
    │   ├── S001C001P001R001A004.skeleton
    │   ├── S001C001P001R001A005.skeleton
    │   ├── S001C001P001R001A006.skeleton
    │   ├── S001C001P001R001A007.skeleton
    │   ├── ....
    │   └── S017C003P020R002A060.skeleton
    ├── denoised_data
    │   ├── actors_info
    │   │   ├── S001C001P001R001A024.txt
    │   │   ├── S001C001P001R001A025.txt
    │   │   ├── S001C001P001R001A026.txt
    │   │   ├── ....
    │   │   ├── S017C003P020R002A059.txt
    │   │   └── S017C003P020R002A060.txt
    │   ├── denoised_failed_1.log
    │   ├── denoised_failed_2.log
    │   ├── frames_cnt.txt
    │   ├── missing_skes_1.log
    │   ├── missing_skes_2.log
    │   ├── missing_skes.log
    │   ├── noise_length.log
    │   ├── noise_motion.log
    │   ├── noise_spread.log
    │   ├── raw_denoised_colors.pkl
    │   ├── raw_denoised_joints.pkl
    │   └── rgb+ske
    ├── raw_data
    │   ├── frames_cnt.txt
    │   ├── frames_drop.log
    │   ├── frames_drop_skes.pkl
    │   └── raw_skes_data.pkl
    ├── get_raw_denoised_data.py
    ├── get_raw_skes_data.py
    ├── seq_transformation.py
    ├── statistics
    │   ├── camera.txt
    │   ├── label.txt
    │   ├── performer.txt
    │   ├── replication.txt
    │   ├── setup.txt
    │   └── skes_available_name.txt
    ├── xview
    │   ├── train_data.npy
    │   ├── train_label.pkl
    │   ├── val_data.npy
    │   └── val_label.pkl
    └── xsub
        ├── train_data.npy
        ├── train_label.pkl
        ├── val_data.npy
        └── val_label.pkl
```

> Note：dictory `denoised_data`、`raw_data`and`nturgb+d_skeletons`, that are temporal files, can be deleted, if extracted `xview` and `xsub`.
