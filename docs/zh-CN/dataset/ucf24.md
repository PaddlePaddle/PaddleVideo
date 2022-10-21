简体中文 | [English](../../en/dataset/ucf24.md)

# UCF24数据准备
UCF24数据的相关准备。主要包括UCF24的RGB帧文件、标注文件的下载和生成文件的路径list。

---
## 1. 数据下载
UCF24数据的详细信息可以参考网站[UCF24](http://www.thumos.info/download.html)。 为了方便使用，PaddleVideo提供了UCF24数据的RGB帧、标注文件的下载脚本。

首先，请确保在[data/ucf24/ 目录](../../../data/ucf24)下，输入如下UCF24数据集的RGB帧、标注文件的命令。

```shell
bash download_frames_annotations.sh
```

- 运行该命令需要安装unrar解压工具，可使用pip方式安装。

- RGB帧文件会存储在[data/ucf24/rgb-images/ 文件夹](../../../data/ucf24/rgb-images)下

- 标注文件会存储在[data/ucf24/lables/ 文件夹](../../../data/ucf24/labels)下

---
## 2. 生成文件的路径list
指定格式划分文件，输入如下命令

```python
python build_split.py --raw_path ./splitfiles
```

**参数说明**

`--raw_path`： 表示原始划分文件的存储路径


# 以上步骤完成后，文件组织形式如下所示

```
├── data
│   ├── ucf24
│   |   ├── groundtruths_ucf
│   |   ├── labels
│   |   |   ├── Basketball
│   |   |   |   ├── v_Basketball_g01_c01
│   |   |   |   |   ├── 00009.txt
│   |   |   |   |   ├── 00010.txt
│   |   |   |   |   ├── ...
│   |   |   |   |   ├── 00050.txt
│   |   |   |   |   ├── 00051.txt
│   |   |   ├── ...
│   |   |   ├── WalkingWithDog
│   |   |   |   ├── v_WalkingWithDog_g01_c01
│   |   |   |   ├── ...
│   |   |   |   ├── v_WalkingWithDog_g25_c04
│   |   ├── rgb-images
│   |   |   ├── Basketball
│   |   |   |   ├── v_Basketball_g01_c01
│   |   |   |   |   ├── 00001.jpg
│   |   |   |   |   ├── 00002.jpg
│   |   |   |   |   ├── ...
│   |   |   |   |   ├── 00140.jpg
│   |   |   |   |   ├── 00141.jpg
│   |   |   ├── ...
│   |   |   ├── WalkingWithDog
│   |   |   |   ├── v_WalkingWithDog_g01_c01
│   |   |   |   ├── ...
│   |   |   |   ├── v_WalkingWithDog_g25_c04
│   |   ├── splitfiles
│   |   |   ├── trainlist01.txt
│   |   |   |── testlist01.txt 
│   |   ├── trainlist.txt
│   |   |── testlist.txt 
```
