[Simplified Chinese](../../../zh-CN/model_zoo/estimation/adds.md) | English

# ADDS-DepthNet model

## content

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Reference](#Reference)

Before getting started, you need to install additional dependencies as follows:
```bash
python -m pip install scikit-image
python -m pip install matplotlib
```

## Introduction

This model is based on the ICCV 2021 paper **[Self-supervised Monocular Depth Estimation for All Day Images using Domain Separation](https://arxiv.org/abs/2108.07628)** of Baidu Robotics and Autonomous Driving Laboratory,
The self-supervised monocular depth estimation model based on day and night images is reproduced, which utilizes the complementary nature of day and night image data, and slows down the large domain shift of day and night images and the accuracy of depth estimation caused by lighting changes. Impact, the most advanced depth estimation results of all-sky images have been achieved on the challenging Oxford RobotCar data set.


## Data

For data download and preparation of Oxford RobotCar dataset, please refer to [Oxford RobotCar dataset data preparation](../../dataset/Oxford_RobotCar.md)


## Train

### Oxford RobotCar dataset training

#### Download and add pre-trained models

1. Download the image pre-training model [resnet18.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/Resnet18_Imagenet.pdparams) as Backbone initialization parameters, or download through the wget command

   ```bash
   wget -P ./data https://videotag.bj.bcebos.com/PaddleVideo-release2.2/Resnet18_Imagenet.pdparams
   ```

2. Open `PaddleVideo/configs/estimation/adds/adds.yaml`, and fill in the downloaded weight storage path below `pretrained:`

    ```yaml
    MODEL: #MODEL field
        framework: "DepthEstimator" #Mandatory, indicate the type of network, associate to the'paddlevideo/modeling/framework/'.
        backbone: #Mandatory, indicate the type of backbone, associate to the'paddlevideo/modeling/backbones/'.
            name: 'ADDS_DepthNet'
            pretrained: fill in the path here
    ```

#### Start training

- The Oxford RobotCar dataset uses a single card for training, and the starting command for the training method is as follows:

    ```bash
    python3.7 main.py --validate -c configs/estimation/adds/adds.yaml --seed 20
    ```


## Test

- The ADDS-DepthNet model is verified synchronously during training (only the day or night data is verified). You can find the keyword `best` in the training log to obtain the model test accuracy. The log example is as follows:

  ```bash
  Already save the best model (rmse)8.5531
  ```

- Because the model can only test one day or night data set at a given path in the yaml file at a time, to get the complete test score at the beginning of this document, you need to run 4 test commands and record their indicators ( 40m during the day, 60m during the day, 40m at night, 60m at night)

- Download URL of the trained model: [ADDS_car.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ADDS_car.pdparams)

- The test commands are as follows:

  ```bash
  # Night 40m
  python3.7 main.py --test -c configs/estimation/adds/adds.yaml -w "output/ADDS/ADDS_best.pdparams" -o DATASET.test.file_path="data/oxford/splits/oxford_day/val_night_files.txt" -o MODEL.head.max_gt_depth=40

  # Night 60m
  python3.7 main.py --test -c configs/estimation/adds/adds.yaml -w "output/ADDS/ADDS_best.pdparams" -o DATASET.test.file_path="data/oxford/splits/oxford_day/val_night_files.txt" -o MODEL.head.max_gt_depth=60

  # Daytime 40m
  python3.7 main.py --test -c configs/estimation/adds/adds.yaml -w "output/ADDS/ADDS_best.pdparams" -o DATASET.test.file_path="data/oxford/splits/oxford_day/val_day_files.txt" -o MODEL.head.max_gt_depth=40

  # Daytime 60m
  python3.7 main.py --test -c configs/estimation/adds/adds.yaml -w "output/ADDS/ADDS_best.pdparams" -o DATASET.test.file_path="data/oxford/splits/oxford_day/val_day_files.txt" -o MODEL.head.max_gt_depth=60
  ```

    The test indicators on the validation dataset of Oxford RobotCar dataset are as follows:

  | version | Max Depth | Abs Rel | Sq Rel | RMSE | RMSE log | <img src="https://latex.codecogs.com/svg.image?\delta&space;<&space;1.25&space;" title="\delta < 1.25 " /> | <img src="https://latex.codecogs.com/svg.image?\delta&space;<&space;1.25^2" title="\delta < 1.25^2" /> | <img src="https://latex.codecogs.com/svg.image?\delta&space;<&space;1.25^3" title="\delta < 1.25^3" /> |
  | ----------- | --------- | ------- | ------ | ----- | ------- | ----------------- |------------------- | ------------------- |
  | ours(night) | 40 | 0.209 | 1.741 | 6.031 | 0.243 | 0.708 | 0.923 | 0.975 |
  | ours(night) | 60 | 0.207 | 2.052 | 7.888 | 0.258 | 0.686 | 0.909 | 0.970 |
  | ours(day) | 40 | 0.114 | 0.574 | 3.411 | 0.157 | 0.860 | 0.977 | 0.993 |
  | ours(day) | 60 | 0.119 | 0.793 | 4.842 | 0.173 | 0.838 | 0.967 | 0.991 |

## Inference

### Export inference model

```bash
python3.7 tools/export_model.py -c configs/estimation/adds/adds.yaml -p data/ADDS_car.pdparams -o inference/ADDS
```

The above command will generate the model structure file `ADDS.pdmodel` and model weight files `ADDS.pdiparams` and `ADDS.pdiparams.info` files needed for prediction, all of which are stored in the `inference/ADDS/` directory

For the meaning of each parameter in the above bash command, please refer to [Model Inference Method](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/en/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### Use predictive engine inference

```bash
python3.7 tools/predict.py --input_file data/example.png \
                           --config configs/estimation/adds/adds.yaml \
                           --model_file inference/ADDS/ADDS.pdmodel \
                           --params_file inference/ADDS/ADDS.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

At the end of the inference, the depth map estimated by the model will be saved in pseudo-color by default.

The following is a sample picture and the corresponding predicted depth map：

<img src="../../../images/oxford_image.png" width = "512" height = "256" alt="image" align=center />

<img src="../../../images/oxford_image_depth.png" width = "512" height = "256" alt="depth" align=center />


## Reference

- [Self-supervised Monocular Depth Estimation for All Day Images using Domain Separation](https://arxiv.org/abs/2108.07628), Liu, Lina and Song, Xibin and Wang, Mengmeng and Liu, Yong and Zhang, Liangjun
