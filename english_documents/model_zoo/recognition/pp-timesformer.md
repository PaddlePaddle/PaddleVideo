[简体中文](../../../zh-CN/model_zoo/recognition/pp-timesformer.md) | English

# TimeSformer Video Classification Model

## Content

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Reference](#Reference)


## Introduction

We have improved the [TimeSformer model](./timesformer.md) and obtained a more accurate 2D practical video classification model **PP-TimeSformer**. Without increasing the amount of parameters and calculations, the accuracy on the UCF-101, Kinetics-400 and other data sets significantly exceeds the original version. The accuracy on the Kinetics-400 data set is shown in the table below.

| Version | Top1 |
| :------ | :----: |
| Ours ([swa](#refer-anchor-1)+distill+16frame) | 79.44 |
| Ours ([swa](#refer-anchor-1)+distill)  | 78.87 |
| Ours ([swa](#refer-anchor-1)) | **78.61** |
| [mmaction2](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/timesformer#kinetics-400) | 77.92 |


## Data

K400 data download and preparation please refer to [Kinetics-400 data preparation](../../dataset/k400.md)

UCF101 data download and preparation please refer to [UCF-101 data preparation](../../dataset/ucf101.md)


## Train

### Kinetics-400 data set training

#### Download and add pre-trained models

1. Download the image pre-training model [ViT_base_patch16_224_miil_21k.pdparams](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams) as Backbone initialization parameters, or download through wget command

   ```bash
   wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams
   ```

2. Open `PaddleVideo/configs/recognition/pptimesformer/pptimesformer_k400_videos.yaml`, and fill in the downloaded weight storage path below `pretrained:`

    ```yaml
    MODEL:
        framework: "RecognizerTransformer"
        backbone:
            name: "VisionTransformer_tweaks"
            pretrained: fill in the path here
    ```

#### Start training

- The Kinetics400 data set uses 8 cards for training, and the start command of the training method is as follows:

    ```bash
    # videos data format
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_pptimesformer main.py --validate -c configs/recognition/ pptimesformer/pptimesformer_k400_videos.yaml
    ```

- Turn on amp mixed-precision training to speed up the training process. The training start command is as follows:

    ```bash
    export FLAGS_conv_workspace_size_limit=800 # MB
    export FLAGS_cudnn_exhaustive_search=1
    export FLAGS_cudnn_batchnorm_spatial_persistent=1
    # videos data format
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_pptimesformer main.py --amp --validate -c configs /recognition/pptimesformer/pptimesformer_k400_videos.yaml
    ```

- In addition, you can customize and modify the parameter configuration to achieve the purpose of training/testing on different data sets. It is recommended that the naming method of the configuration file is `model_dataset name_file format_data format_sampling method.yaml` , Please refer to [config](../../tutorials/config.md) for parameter usage.


## Test

- The PP-TimeSformer model is verified synchronously during training. You can find the keyword `best` in the training log to obtain the model test accuracy. The log example is as follows:

  ```
  Already save the best model (top1 acc)0.7258
  ```

- Because the sampling method of the PP-TimeSformer model test mode is a slightly slower but higher accuracy **UniformCrop**, which is different from the **RandomCrop** used in the verification mode during the training process, so the verification index recorded in the training log` topk Acc` does not represent the final test score, so after the training is completed, you can use the test mode to test the best model to obtain the final index. The command is as follows:

  ```bash
  # 8-frames testing script
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptimesformer  main.py  --test -c configs/recognition/pptimesformer/pptimesformer_k400_videos.yaml -w "output/ppTimeSformer/ppTimeSformer_best.pdparams"

  # 16-frames testing script
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptimesformer main.py --test \
  -c configs/recognition/pptimesformer/pptimesformer_k400_videos.yaml \
  -o MODEL.backbone.num_seg=16 \
  -o MODEL.runtime_cfg.test.num_seg=16 \
  -o PIPELINE.test.decode.num_seg=16 \
  -o PIPELINE.test.sample.num_seg=16 \
  -w "data/ppTimeSformer_k400_16f_distill.pdparams"
  ```


  When the test configuration uses the following parameters, the test indicators on the validation data set of Kinetics-400 are as follows:

   | backbone           | Sampling method | num_seg | target_size | Top-1 | checkpoints |
   | :----------------: | :-------------: | :-----: | :---------: | :---- | :----------------------------------------------------------: |
   | Vision Transformer |   UniformCrop   |   8    |     224     | 78.61 | [ppTimeSformer_k400_8f.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTimeSformer_k400_8f.pdparams) |
   | Vision Transformer | UniformCrop | 8 | 224 | 78.87 | [ppTimeSformer_k400_8f_distill.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTimeSformer_k400_8f_distill.pdparams) |
   | Vision Transformer | UniformCrop | 16 | 224 | 79.44 | [ppTimeSformer_k400_16f_distill.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTimeSformer_k400_16f_distill.pdparams) |


- During the test, the PP-TimeSformer video sampling strategy is to use linspace sampling: in time sequence, from the first frame to the last frame of the video sequence to be sampled, `num_seg` sparse sampling points (including endpoints) are uniformly generated; spatially , Select 3 areas to sample at both ends of the long side and the middle position (left, middle, right or top, middle, and bottom). A total of 1 clip is sampled for 1 video.

## Inference

### Export inference model

```bash
python3.7 tools/export_model.py -c configs/recognition/pptimesformer/pptimesformer_k400_videos.yaml \
                                -p data/ppTimeSformer_k400_8f.pdparams \
                                -o inference/ppTimeSformer
```

The above command will generate the model structure file `ppTimeSformer.pdmodel` and the model weight file `ppTimeSformer.pdiparams` required for prediction.

- For the meaning of each parameter, please refer to [Model Reasoning Method](../../start.md#2-Model Reasoning)

### Use predictive engine inference

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/pptimesformer/pptimesformer_k400_videos.yaml \
                           --model_file inference/ppTimeSformer/ppTimeSformer.pdmodel \
                           --params_file inference/ppTimeSformer/ppTimeSformer.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

The output example is as follows:

```
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 0.9997474551200867
```

It can be seen that using the ppTimeSformer model trained on Kinetics-400 to predict `data/example.avi`, the output top1 category id is `5`, and the confidence is 0.99. By referring to the category id and name correspondence table `data/k400/Kinetics-400_label_list.txt`, it can be known that the predicted category name is `archery`.

## Reference

- [Is Space-TimeAttention All You Need for Video Understanding?](https://arxiv.org/pdf/2102.05095.pdf), Gedas Bertasius, Heng Wang, Lorenzo Torresani
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), Geoffrey Hinton, Oriol Vinyals, Jeff Dean
<div id="refer-anchor-1"></div>

- [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407v3), Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov
- [ImageNet-21K Pretraining for the Masses](https://arxiv.org/pdf/2104.10972v4.pdf), Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy
