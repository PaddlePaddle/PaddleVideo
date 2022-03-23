[Simplified Chinese](../../../zh-CN/model_zoo/resolution/wafp.md)) | English

# WAFP-Net model

## content

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Reference](#Reference)

Before getting started, you need to install additional dependencies by following the command:

```bash
python -m pip install scipy
python -m pip install h5py
```

## Introduction

This model is based on the **IEEE Transactions on Multimedia 2021 paper of Baidu Robotics and Autonomous Driving Laboratory [WAFP-Net: Weighted Attention Fusion based Progressive Residual Learning for Depth Map Super-resolution](https://ieeexplore.ieee.org/document/9563214/)** for reference.
A depth map super-resolution model based on adaptive fusion attention is reproduced, and an adaptive fusion attention is proposed for two image degradation methods (interval sampling and noisy bicubic sampling) existing in real scenes. This mechanism achieves state-of-the-art accuracy on multiple datasets while maintaining the superiority of model parameters.


## Data

### Mixed Dataset

The data used in this document combines three datasets, Middlebury dataset/ MPI Sintel dataset and synthetic New Tsukuba dataset
1. Prepare raw image data

    Download these two compressed package: https://videotag.bj.bcebos.com/Data/WAFP_data.zip,https://videotag.bj.bcebos.com/Data/WAFP_test_data.zip
    Unzip them, and place the `data_all` folder(containing 133 depth maps) and `test_data`(containing 4 test mat) in the following locations:

    ```shell
    data/
    └── depthSR/
        ├── data_all/
        │   ├── alley_1_1.png
        │   ├── ...
        │   └── ...
        ├── test_data/
        │   ├── cones_x4.mat
        │   ├── teddy_x4.mat
        │   ├── tskuba_x4.mat
        │   └── venus_x4.mat
        ├── val.list
        ├── generate_train_noise.m
        └── modcrop.m
    ```

2. Execute the `generate_train_noise.m` script to generate the training data `train_depth_x4_noise.h5`, and use the `ls` command to generate the `val.list` path file.
    ```shell
    cd data/depthSR/
    generate_train_noise.m
    ls test_data > test.list
    cd ../../
    ```

3. Fill in the paths of `train_depth_x4_noise.h5`, `test_data`, and `test.list` to the corresponding positions of `wafp.yaml`:
    ```yaml
    DATASET: #DATASET field
    batch_size: 64 #Mandatory, bacth size
    valid_batch_size: 1
    test_batch_size: 1
    num_workers: 1 #Mandatory, XXX the number of subprocess on each GPU.
    train:
        format: "HDF5Dataset"
        file_path: "data/depthSR/train_depth_x4_noise.h5" # path of train_depth_x4_noise.h5
    valid:
        format: "MatDataset"
        data_prefix: "data/depthSR/test_data" # path of test_data
        file_path: "data/depthSR/test.list" # path of test.list
    test:
        format: "MatDataset"
        data_prefix: "data/sintel/test_data" # path of test_data
        file_path: "data/sintel/test.list" # path of test.list
    ```

## Train

### Mixed dataset training

#### start training

- The Mixed dataset is trained with a single card. The start command of the training method is as follows:

    ```bash
    python3.7 main.py -c configs/resolution/wafp/wafp.yaml --seed 42
    ```


## Test

- Download address of the trained model: [WAFP.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/WAFP_best.pdparams)

- The test command is as follows:

  ```bash
  python3.7 main.py --test -c configs/resolution/wafp/wafp.yaml -w "output/WAFP/WAFP_epoch_00080.pdparams"
  ```

    The test metrics on the validation dataset of the Oxford RobotCar dataset are as follows:

  | version | RMSE    |  SSIM   |
  | :------ | :-----: | :-----: |
  | ours    |  2.5479 |  0.9808 |

## Inference

### Export inference model

```bash
python3.7 tools/export_model.py -c configs/resolution/wafp/wafp.yaml -p data/WAFP.pdparams -o inference/WAFP
```

The above command will generate the model structure file `WAFP.pdmodel` and model weight files `WAFP.pdiparams` and `WAFP.pdiparams.info` files required for prediction, which are stored in the `inference/WAFP/` directory

For the meaning of each parameter in the above bash command, please refer to [Model Inference Method](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)

### Inference using the prediction engine

```bash
python3.7 tools/predict.py --input_file data/example.mat \
--config configs/resolution/wafp/wafp.yaml \
--model_file inference/WAFP/WAFP.pdmodel \
--params_file inference/WAFP/WAFP.pdiparams \
--use_gpu=True \
--use_tensorrt=False
```

At the end of the inference, the depth map output by the model will be saved in pseudo-color by default.

The following are sample images and corresponding predicted depth maps:

<img src="../../../images/cones_x4_wafp_input.png" alt="input" align=center />

<img src="../../../images/cones_x4_wafp_output.png" alt="output" align=center />


## Reference

- [WAFP-Net: Weighted Attention Fusion based Progressive Residual Learning for Depth Map Super-resolution](https://ieeexplore.ieee.org/document/9563214/), Xibin Song, Dingfu Zhou, Wei Li∗, Yuchao Dai, Liu Liu, Hongdong Li, Ruigang Yang and Liangjun Zhang
