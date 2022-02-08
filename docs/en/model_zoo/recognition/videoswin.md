[简体中文](../../../zh-CN/model_zoo/recognition/videoswin.md) | English

# Video-Swin-Transformer Video Classification Model

## content

- [Introduction](#Introduction)
- [Data](#DATA)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Reference](#Reference)


## Introduction

Video-Swin-Transformer is a video classification model based on Swin Transformer. It utilizes Swin Transformer's multi-scale modeling and efficient local attention characteristics. It currently achieves SOTA accuracy on the Kinetics-400 data set, surpassing the same transformer structure. The TimeSformer model.


![VideoSwin](../../../images/videoswin.jpg)

## DATA

K400 data download and preparation please refer to [Kinetics-400 data preparation](../../dataset/k400.md)


## Train

### Kinetics-400 data set training

#### Download and add pre-trained models

1. Download the image pre-training model [SwinTransformer_imagenet.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/SwinTransformer_imagenet.pdparams) as the Backbone initialization parameter, or download it through the wget command

   ```bash
   wget https://videotag.bj.bcebos.com/PaddleVideo-release2.2/SwinTransformer_imagenet.pdparams
   ```

2. Open `configs/recognition/videoswin/videoswin_k400_videos.yaml`, and fill in the downloaded weight storage path below `pretrained:`

    ```yaml
    MODEL:
        framework: "RecognizerTransformer"
        backbone:
            name: "SwinTransformer3D"
            pretrained: fill in the path here
    ```

#### Start training

- The Kinetics400 data set uses 8 cards for training, and the start command of the training method is as follows:

    ```bash
    # videos data format
    python3.7 -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_videoswin main.py --validate -c configs/ recognition/video_swin_transformer/videoswin_k400_videos.yaml
    ```

- Turn on amp mixed-precision training to speed up the training process. The training start command is as follows:

    ```bash
    export FLAGS_conv_workspace_size_limit=800 # MB
    export FLAGS_cudnn_exhaustive_search=1
    export FLAGS_cudnn_batchnorm_spatial_persistent=1
    # videos data format
    python3.7 -u -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_videoswin main.py --amp --validate- c configs/recognition/videoswin/videoswin_k400_videos.yaml
    ```

- In addition, you can customize and modify the parameter configuration to achieve the purpose of training/testing on different data sets. It is recommended that the naming method of the configuration file is `model_dataset name_file format_data format_sampling method.yaml` , Please refer to [config](../../tutorials/config.md) for parameter usage.


## Test

- The Video-Swin-Transformer model is verified during training. You can find the keyword `best` in the training log to obtain the model test accuracy. The log example is as follows:

  ```
  Already save the best model (top1 acc)0.7258
  ```

- Since the sampling method of the Video-Swin-Transformer model test mode is a bit slower but more accurate **UniformCrop**, which is different from the **CenterCrop** used in the verification mode during the training process, so the verification recorded in the training log The index `topk Acc` does not represent the final test score, so after the training is completed, you can use the test mode to test the best model to obtain the final index. The command is as follows:

  ```bash
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_videoswin main.py --test -c configs/recognition/ video_swin_transformer/videoswin_k400_videos.yaml -w "output/VideoSwin/VideoSwin_best.pdparams"
  ```


  When the test configuration uses the following parameters, the test indicators on the validation data set of Kinetics-400 are as follows:

   | backbone | Sampling method | num_seg | target_size | Top-1 | checkpoints |
   | :----------------: | :-------------: | :-----: | :---------: | :---- | :----------------------------------------------------------: |
   | Swin Transformer | UniformCrop | 32 | 224 | 82.40 | [SwinTransformer_k400.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/VideoSwin_k400.pdparams) |


## Inference

### Export inference model

```bash
python3.7 tools/export_model.py -c configs/recognition/videoswin/videoswin_k400_videos.yaml \
                                -p data/VideoSwin_k400.pdparams \
                                -o inference/VideoSwin
```

The above command will generate the model structure file `VideoSwin.pdmodel` and the model weight file `VideoSwin.pdiparams` required for prediction.

- For the meaning of each parameter, please refer to [Model Reasoning Method](../../start.md#2-Model Reasoning)

### Use predictive engine inference

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/videoswin/videoswin_k400_videos.yaml \
                           --model_file inference/VideoSwin/VideoSwin.pdmodel \
                           --params_file inference/VideoSwin/VideoSwin.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

The output example is as follows:

```
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 0.9999829530715942
```

It can be seen that using the Video-Swin-Transformer model trained on Kinetics-400 to predict `data/example.avi`, the output top1 category id is `5`, and the confidence is 0.99. By referring to the category id and name correspondence table `data/k400/Kinetics-400_label_list.txt`, it can be known that the predicted category name is `archery`.

## Reference

- [Video Swin Transformer](https://arxiv.org/pdf/2106.13230.pdf), Ze Liu, Jia Ning, Yue Cao, Yixuan Wei
