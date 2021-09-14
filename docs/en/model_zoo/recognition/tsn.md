[简体中文](../../../zh-CN/model_zoo/recognition/tsn.md) | English

# TSN

## Content

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Details](#Details)
- [Reference](#Reference)

## Introduction

Temporal Segment Network (TSN) is a classic 2D-CNN-based solution in the field of video classification. This method mainly solves the problem of long-term behavior recognition of video, and replaces dense sampling by sparsely sampling video frames, which can not only capture the global information of the video, but also remove redundancy and reduce the amount of calculation. The core idea is to average the features of each frame as the overall feature of the video, and then enter the classifier for classification. The model implemented by this code is a TSN network based on a single-channel RGB image, and Backbone uses the ResNet-50 structure.

<div align="center">
<img src="../../../images/tsn_architecture.png" height=350 width=80000 hspace='10'/> <br />
</div>


For details, please refer to the ECCV 2016 paper [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859)

## Data

PaddleVide provides training and testing scripts on the Kinetics-400 dataset. Kinetics-400 data download and preparation please refer to [Kinetics-400 data preparation](../../dataset/k400.md)

## Train

### Kinetics-400 data set training

#### Download and add pre-trained models

1. Load the ResNet50 weights trained on ImageNet1000 as Backbone initialization parameters [ResNet50_pretrain.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams), or download through the command line

   ```bash
   wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams
   ```

2. Open `PaddleVideo/configs/recognition/tsn/tsn_k400_frames.yaml`, and fill in the downloaded weight path below `pretrained:`

   ```yaml
   MODEL:
       framework: "Recognizer2D"
       backbone:
           name: "ResNet"
           pretrained: fill in the path here
   ```

#### Start training

- Kinetics-400 data set uses 8 cards for training, the training start command for frames format data is as follows

  ```bash
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_tsn main.py --validate -c configs/recognition/ tsn/tsn_k400_frames.yaml
  ```

## Test

Since the sampling method of the TSN model test mode is **TenCrop** with a slower speed but higher accuracy, which is different from the **CenterCrop** used in the verification mode during the training process, the verification index `topk Acc` recorded in the training log It does not represent the final test score, so after the training is completed, you can use the test mode to test the best model to obtain the final index. The command is as follows:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_tsn main.py --test -c configs/recognition/ tsn/tsn_k400_frames.yaml -w "output/TSN/TSN_best.pdparams"
```

When the test configuration uses the following parameters, the test indicators on the validation data set of Kinetics-400 are as follows:

| backbone | Sampling method | Training Strategy | num_seg | target_size | Top-1 |                         checkpoints                          |
| :------: | :-------------: | :---------------: | :-----: | :---------: | :---: | :----------------------------------------------------------: |
| ResNet50 |     TenCrop     |       NCHW        |   3    |     224     | 69.81 | [TSN_k400.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TSN_k400.pdparams) |
| ResNet50 |     TenCrop     |       NCHW        |   8    |     224     | 71.70 | [TSN_k400_8.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/TSN_k400_8.pdparams) |
## Inference

### export inference model

```bash
python3.7 tools/export_model.py -c configs/recognition/tsn/tsn_k400_frames.yaml \
                                -p data/TSN_k400.pdparams \
                                -o inference/TSN
```

The above command will generate the model structure file `TSN.pdmodel` and the model weight file `TSN.pdiparams` required for prediction.

For the meaning of each parameter, please refer to [Model Reasoning Method](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-Model Reasoning)

### infer

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/tsn/tsn_k400_frames.yaml \
                           --model_file inference/TSN/TSN.pdmodel \
                           --params_file inference/TSN/TSN.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```
- **Note**: For models that combine N and T during calculation (such as TSN, TSM), when `use_tensorrt=True`, you need to specify the `batch_size` argument as batch_size\*num_seg.

    ```bash
    python3.7 tools/predict.py --input_file data/example.avi \
                               --config configs/recognition/tsn/tsn_k400_frames.yaml \
                               --model_file inference/TSN/TSN.pdmodel \
                               --params_file inference/TSN/TSN.pdiparams \
                               --batch_size 3 \
                               --use_gpu=True \
                               --use_tensorrt=True
    ```
## Details

**data processing:**

- The model reads the `mp4` data in the Kinetics-400 data set, first divides each piece of video data into `seg_num` segments, and then evenly extracts 1 frame of image from each segment to obtain sparsely sampled `seg_num` video frames , And then do the same random data enhancement to this `seg_num` frame image, including multi-scale random cropping, random left and right flips, data normalization, etc., and finally zoom to `target_size`

**training strategy:**

- Use Momentum optimization algorithm for training, momentum=0.9

- Using L2_Decay, the weight attenuation coefficient is 1e-4

- Use global gradient clipping, with a clipping factor of 40.0

- The total number of epochs is 100, and the learning rate will be attenuated by 0.1 times when the epoch reaches 40 and 80

- Dropout_ratio=0.4

**parameter initialization**

- The convolutional layer of the TSN model uses Paddle's default [KaimingNormal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/initializer/KaimingNormal_cn.html#kaimingnormal) and [Constant](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/nn/initializer/Constant_cn.html#constant) initialization method, with Normal(mean=0, std= 0.01) normal distribution to initialize the weight of the FC layer, and a constant 0 to initialize the bias of the FC layer

## Reference

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/abs/1608.00859), Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool
