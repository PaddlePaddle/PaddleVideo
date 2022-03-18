[English](../../en/dataset/ntu-rgbd.md) | 简体中文

# NTU-RGB+D 数据准备

- [数据集介绍](#数据集介绍)
- [ST-GCN数据集准备](#ST-GCN数据集准备)
- [CTR-GCN数据集准备](#CTR-GCN数据集准备)

---


## 数据集介绍

NTU-RGB+D是基于骨骼的行为识别数据集，包含60个种类的动作，56880个样本，详细介绍可以参考其官方网站[NTU-RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/)。该数据集在划分训练集和测试集时采用了两种不同的划分标准。Cross-Subject按照人物ID划分，训练集40320个样本，测试集16560个样本。Cross-View安装相机划分，相机2和3采集的样本为训练集，包含37930个样本，相机1采集的样本为测试集，包含18960个样本。


## ST-GCN数据集准备

以下是ST-GCN模型的数据集准备流程介绍。

### 数据集下载

我们提供处理好的数据集下载地址[NTU-RGB-D.tar](https://videotag.bj.bcebos.com/Data/NTU-RGB-D.tar)(~3.1G)，下载后通过命令```tar -zxvf NTU-RGB-D.tar ```进行解压，得到的数据目录如下：

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

> 数据来源于[st-gcn](https://github.com/open-mmlab/mmskeleton/blob/master/doc/SKELETON_DATA.md)。

## CTR-GCN数据集准备

以下是CTR-GCN模型的数据集准备流程介绍。

### 数据集下载

在`data\ntu-rgb-d`目录有下载其官方网站[NTU-RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/)提供的数据集的脚本`download_dataset.sh`

```bash
sh data/ntu-rgb-d/download_dataset.sh
```

运行脚本后会得到如下的数据目录：
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

### 数据集处理

运行如下脚本，将数据处理成CTR-GCN所需的格式。

> 注：若自定义数据集，提前准备好`data/ntu-rgb-d/statistics/skes_available_name.txt`文件，该文件是待处理的骨骼点数据文件名清单。

```bash
cd ./data/ntu-rgb-d
# Get skeleton of each performer
python get_raw_skes_data.py
# Remove the bad skeleton
python get_raw_denoised_data.py
# Transform the skeleton to the center of the first frame
python seq_transformation.py
```

最终数据集处理后得到如下文件树形式

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

> 注：文件夹`denoised_data`、`raw_data`和`nturgb+d_skeletons`都为处理处理的临时文件，可在提取出`xview`和`xsub`后删除。
