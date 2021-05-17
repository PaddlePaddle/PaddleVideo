- [简体中文](../../../zh-CN/model_zoo/recognition/tsm.md) | English

  # TSM

  ## Contents

  - [Introduction](#Introduction)
  - [Data](#Data)
  - [Train](#Train)
  - [Test](#Test)
  - [Inference](#Inference)
  - [Reference](#Reference)


  ## Introduction

  Temporal Shift Module (TSM) is a popular model that attracts more attention at present.
  The method of moving through channels greatly improves the utilization ability of temporal information without increasing any additional number of parameters and calculation amount.
  Moreover, due to its lightweight and efficient characteristics, it is very suitable for industrial landing.

  <div align="center">
  <img src="../../../images/tsm_architecture.png" height=250 width=700 hspace='10'/> <br />
  </div>


  This code implemented **single RGB stream** of TSM networks. Backbone is ResNet-50.

  Please refer to the ICCV 2019 paper for details [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf)

  ## Data

  Please refer to K400 data download and preparation [k400 data preparation](../../dataset/K400.md)

  Please refer to UCF101 data download and preparation [ucf101 data preparation](../../dataset/ucf101.md)


  ## Train

  ### download pretrain-model

  1. Please download [ResNet50_pretrain.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams) as pretraind model:

     ```bash
     wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_pretrain.pdparams
     ```

  2. and add path to MODEL.framework.backbone.pretrained in config file as:

     ```yaml
     MODEL:
         framework: "Recognizer2D"
         backbone:
             name: "ResNetTSM"
             pretrained: your weight path
     ```

  ### Start training

  - By specifying different configuration files, different data formats/data sets can be used for training. Taking the training configuration of Kinetics-400 data set + 8 cards + frames format as an example, the startup command is as follows (more training commands can be viewed in `PaddleVideo/run.sh`).

    ```bash
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7,8" --log_dir=log_tsm main.py  --validate -c configs/recognition/tsm/tsm_k400_frames.yaml
    ```

  - Args -c is used to specify config file.

  - For finetune (take the training configuration of UCF-101 + 4 cards + frames format as an example), please download the published model weights of PaddleVideo first [TSM.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/TSM/TSM.pdparams), then replace the field after `pretrained:` in `PaddleVideo/configs/recognition/tsm/tsm_ucf101_frames.yaml` with the downloaded path of `TSM.pdparams`, and finally execute the following command to start finetune training.

    ```bash
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_tsm main.py  --validate -c configs/recognition/tsm/tsm_ucf101_frames.yaml
    ```

  - If you want to continue training, specify the `--weights` parameter as the model to continue training.

    ```bash
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_tsm main.py  --validate -c configs/recognition/tsm/tsm_ucf101_frames.yaml --weights resume_model.pdparams
    ```

  - For the config file usage，please refer to [config](../../tutorials/config.md).

  ### Implementation details

  **data processing**

  - The model reads the `mp4` data in the Kinetics-400 data set, first divides each piece of video data into `seg_num` segments, and then uniformly extracts 1 frame of image from each segment to obtain sparsely sampled `seg_num` video frames. Then do the same random data enhancement to this `seg_num` frame image, including multi-scale random cropping, random left and right flips, data normalization, etc., and finally zoom to `target_size`.

  **Training strategy**

  *  Use Momentum optimization algorithm training, momentum=0.9
  *  Using L2_Decay, the weight attenuation coefficient is 1e-4
  *  Using global gradient clipping, the clipping factor is 20.0
  *  The total number of epochs is 50, and the learning rate will be attenuated by 0.1 times when the epoch reaches 20 and 40
  *  The learning rate of the weight and bias of the FC layer are respectively 5 times and 10 times the overall learning rate, and the bias does not set L2_Decay
  *  Dropout_ratio=0.5

  **Parameter initialization**

  - Initialize the weight of the FC layer with the normal distribution of $N(0,0.001)$, and initialize the bias of the FC layer with a constant of 0

  ## Test

  Put the weight of the model to be tested into the `output/TSM/` directory, the test command is as follows

  ```bash
  python3 main.py --test -c configs/recognition/tsm/tsm.yaml -w output/TSM/TSM_best.pdparams
  ```

  - You can also use our trained and published model [TSM.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/TSM/TSM.pdparams) to test


  Accuracy on Kinetics400:

  | seg\_num | target\_size | Top-1  |
  | :------: | :----------: | :----: |
  |    8     |     224      | 0.7106 |

  ## Inference

  ### export inference model

   To get model architecture file `TSM.pdmodel` and parameters file `TSM.pdiparams`, use:

  ```bash
  python3 tools/export_model.py -c configs/recognition/tsm/tsm_k400_frames.yaml \
                                -p output/TSM/TSM_best.pdparams \
                                -o inference/TSM
  ```

  - Args usage please refer to [Model Inference](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86).

  ### infer

  ```bash
  python3 tools/predict.py --video_file data/example.avi \
                           --model_file inference/TSM/TSM.pdmodel \
                           --params_file inference/TSM/TSM.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
  ```

  ## Reference

  - [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
