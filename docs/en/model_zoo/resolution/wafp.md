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

This model is based on the **IEEE Transactions on Multimedia 2021 paper of Baidu Robotics and Autonomous Driving Laboratory [WAFP-Net: Weighted Attention Fusion based Progressive Residual Learning for Depth Map Super-resolution](https://arxiv.org/abs/2108.07628 )** for reference.
A depth map super-resolution model based on adaptive fusion attention is reproduced, and an adaptive fusion attention is proposed for two image degradation methods (interval sampling and noisy bicubic sampling) existing in real scenes. This mechanism achieves state-of-the-art accuracy on multiple datasets while maintaining the superiority of model parameters.


## Data

TODO


## Train

### Oxford RobotCar dataset training

#### start training

- The Oxford RobotCar dataset is trained with a single card. The start command of the training method is as follows:

    ```bash
    python3.7 main.py -c configs/resolution/wafp/wafp.yaml --seed 42
    ```


## Test

- Download address of the trained model: [WAFP.pdparams](TODO)

- The test command is as follows:

  ```bash
  python3.7 main.py --test -c configs/resolution/wafp/wafp.yaml -w "output/WAFP/WAFP_epoch_00080.pdparams"
  ```

    The test metrics on the validation dataset of the Oxford RobotCar dataset are as follows:

  | version | RMSE    |  SSIM   |
  | :------ | :-----: | :-----: |
  | ours    | 2.5762  |  0.9813 |

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

<img src="../../../images/oxford_image.png" width = "512" height = "256" alt="image" align=center />

<img src="../../../images/oxford_image_depth.png" width = "512" height = "256" alt="depth" align=center />


## Reference

- [WAFP-Net: Weighted Attention Fusion based Progressive Residual Learning for Depth Map Super-resolution](https://arxiv.org/abs/2108.07628), Xibin Song, Dingfu Zhou, Wei Li∗, Yuchao Dai, Liu Liu, Hongdong Li, Ruigang Yang and Liangjun Zhang
