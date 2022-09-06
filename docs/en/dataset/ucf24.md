English | [简体中文](../../zh-CN/dataset/ucf24.md)

# UCF24 Data Preparation
This document mainly introduces the preparation process of UCF24 dataset. It mainly includes the download of the RGB frame files, the annotation files and the pathlist of the generated file.

---
## 1. Data Download
Detailed information on UCF24 data can be found on the website [UCF24](http://www.thumos.info/download.html). For ease of use, PaddleVideo provides a download script for the RGB frame, annotation file of the UCF24 data.

First, please ensure access to the [data/ucf24/ directory](../../../data/ucf24) and enter the following command for downloading the RGB frame, annotation file of the UCF24 dataset.

```shell
bash download_frames_annotations.sh
```

- To run this command you need to install the unrar decompression tool, which can be installed using the pip method.

- The RGB frame files will be stored in the [data/ucf24/rgb-images/ directory](../../../data/ucf24/rgb-images)

- The annotation files will be stored in the [data/ucf24/lables/ directory](../../../data/ucf24/labels)

---
## 2. File Pathlist Generation
To specify the format for dividing the file, enter the following command

```python
python build_split.py --raw_path ./splitfiles
```

**Description of parameters**

`--raw_path`： indicates the storage path of the original division file


# Folder Structure
After the whole data pipeline for UCF24 preparation, the folder structure will look like:

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
