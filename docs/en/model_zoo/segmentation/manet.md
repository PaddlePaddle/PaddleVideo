[简体中文](../../../zh-CN/model_zoo/partition/manet.md) | English

# Ma-Net

## Contents

- [Introduction](#Introduction)

- [Data](#Data)

- [Train](#Train)

- [Test](#Test)

- [Inference](#Inference)




## Introduction

This is the paddle implementation of the CVPR2020 paper "[Memory aggregation networks for efficient interactive video object segmentation](https://arxiv.org/abs/2003.13246)".

![avatar](../../../images/1836-teaser.gif)

This code currently supports model test and model training on DAVIS  dataset,  and model inference on any given video will be provided in few days.



## Data

Download [DAVIS2017](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) and [scribbles](https://data.vision.ee.ethz.ch/csergi/share/DAVIS-Interactive/DAVIS-2017-scribbles-trainval.zip) into one folder. Please refer to [DAVIS](https://davischallenge.org/davis2017/code.html).

If you need the file "DAVIS2017/ImageSets/2017/v_a_l_instances.txt", please refer to the link https://drive.google.com/file/d/1aLPaQ_5lyAi3Lk3d2fOc_xewSrfcrQlc/view?usp=sharing


## Train

#### Download and add pre-trained models

1. Download [deeplabV3+ model pretrained on COCO](https://drive.google.com/file/d/15temSaxnKmGPvNxrKPN6W2lSsyGfCtTB/view?usp=sharing) to this repo as the Backbone initialization parameter, or download it through wget

   ```bash
   wget https://drive.google.com/file/d/15temSaxnKmGPvNxrKPN6W2lSsyGfCtTB/view?usp=sharing
   ```

2. Open `PaddleVideo/configs/segmentationer/manet_stage1.yaml`, and fill in the downloaded weight storage path below `pretrained:`

   ```yaml
   MODEL: #MODEL field
       framework: "ManetSegment_Stage1"
       backbone:
           name: "DeepLab"
           pretrained: fill in the path here
   ```

#### Start training

- Our training process contains two stage.

  -  You can start training of stage one using one card by such command：

    ```bash
    python main.py -c configs/segmentation/manet_stage1.yaml
    ```
- Then you can start training of stage two  using one card (other training method such as multiple cards or amp mixed-precision is similar to the above) by such command which depends on the model training result of stage one：

  ```bash
    python main.py -c configs/segmentation/manet_stage2.yaml
  ```

- In addition, you can customize and modify the parameter configuration to achieve the purpose of training/testing on different data sets. It is recommended that the naming method of the configuration file is `model_dataset name_file format_data format_sampling method.yaml` , Please refer to [config](../../tutorials/config.md) for parameter usage.



## Test

You can start testing by such command：

```bash
python main.py --test -c configs/segmentation/manet_stage2.yaml -w output/ManetSegment_Stage2/ManetSegment_Stage2_step_100001.pdparams  
```

- You can download [our model](https://drive.google.com/file/d/1JjYNha40rtEYKKKFtDv06myvpxagl5dW/view?usp=sharing) decompress it and specify the path to `METRIC.ground_truth_filename` in config file.

- Args `-w` is used to specifiy the model path，you can download [our model](https://drive.google.com/file/d/1JjYNha40rtEYKKKFtDv06myvpxagl5dW/view?usp=sharing) and decompress it  for evaluation.


Test accuracy in DAVIS2017:

| J@60  |  AUC  |
| :---: | :---: |
| 0.761 | 0.749 |
