[简体中文](../../../zh-CN/model_zoo/recognition/tokenshift_transformer.md) | English

# TimeSformer

## Content

- [Introduction](#Introduction)
- [Data](#DATA)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Reference](#Reference)


## Introduction

Token Shift Transformer is a video classification model based on vision transformer, which shares merits of strong interpretability, high discriminative power on hyper-scale data, and ﬂexibility in processing varying length inputs. Token Shift Module is a novel, zero-parameter, zero-FLOPs operator, for modeling temporal relations within each transformer encoder.

<div align="center">
<img src="../../../images/tokenshift_structure.png">
</div>



## Data

UCF-101 data download and preparation please refer to [UCF-101 data preparation](../../dataset/ucf101.md)


## Train

### Kinetics-400 data set training

#### Download and add pre-trained models

1. Download the image pre-training model [ViT_base_patch16_224](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams) as Backbone initialization parameters, or download through the wget command

   ```bash
   wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams
   ```

2. Open `PaddleVideo/configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml`, and fill in the downloaded weight storage path below `pretrained:`

    ```yaml
    MODEL:
        framework: "RecognizerTransformer"
        backbone:
            name: "VisionTransformer"
            pretrained: fill in the path here
    ```

#### Start training

- The UCF-101 data set uses 1 card for training, and the start command of the training method is as follows:

```bash
# videos data format
python3 main.py -c configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml --validate --seed=1234
```

- Turn on amp mixed-precision training to speed up the training process. The training start command is as follows:

```bash
python3 main.py --amp -c configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml --validate --seed=1234
```

- In addition, you can customize and modify the parameter configuration to achieve the purpose of training/testing on different data sets. It is recommended that the naming method of the configuration file is `model_dataset name_file format_data format_sampling method.yaml` , Please refer to [config](../../tutorials/config.md) for parameter usage.


## Test

- The Token Shift Transformer model is verified synchronously during training. You can find the keyword `best` in the training log to obtain the model test accuracy. The log example is as follows:

  ```
  Already save the best model (top1 acc)0.9201
  ```

- Since the sampling method of the Token Shift Transformer model test mode is **uniform** sampling, which is different from the **dense** sampling used in the verification mode during the training process, so the verification index recorded in the training log is `topk Acc `Does not represent the final test score, so after the training is completed, you can use the test mode to test the best model to obtain the final index, the command is as follows:

  ```bash
  python3 main.py --amp -c configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml --test --seed=1234 -w 'output/TokenShiftVisionTransformer/TokenShiftVisionTransformer_best.pdparams'
  ```


  When the test configuration uses the following parameters, the test indicators on the validation data set of UCF-101 are as follows:


  | backbone | sampling method | num_seg | target_size | Top-1 | checkpoints |
  | :----------------: | :-----: | :-----: | :---------: | :----: | :----------------------------------------------------------: |
  | Vision Transformer | Uniform | 8 | 256 | 92.81 | [TokenShiftTransformer.pdparams](https://drive.google.com/drive/folders/1k_TpAqaJZYJE8C5g5pT9phdyk9DrY_XL?usp=sharing) |


- Uniform sampling: Timing-wise, equal division into `num_seg` segments, 1 frame sampled at the middle of each segment; spatially, sampling at the center. 1 video sampled 1 clip in total.

## Inference

### Export inference model

```bash
python3 tools/export_model.py -c configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml -p 'output/TokenShiftVisionTransformer/TokenShiftVisionTransformer_best.pdparams'
```

The above command will generate the model structure file `TokenShiftVisionTransformer.pdmodel` and the model weight file `TokenShiftVisionTransformer.pdiparams` required for prediction.

- For the meaning of each parameter, please refer to [Model Reasoning Method](../../usage.md#2-infer)

### Use prediction engine inference

```bash
python3 tools/predict.py -c configs/recognition/token_transformer/tokShift_transformer_ucf101_256_videos.yaml -i 'data/BrushingTeeth.avi' --model_file ./inference/TokenShiftVisionTransformer.pdmodel --params_file ./inference/TokenShiftVisionTransformer.pdiparams
```

The output example is as follows:

```
Current video file: data/BrushingTeeth.avi
	top-1 class: 19
	top-1 score: 0.9959074258804321
```

It can be seen that using the TimeSformer model trained on Kinetics-400 to predict `data/BrushingTeeth.avi`, the output top1 category id is `19`, and the confidence is 0.99. By consulting the category id and name correspondence table, it can be seen that the predicted category name is `brushing_teeth`.

## Reference

- [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/pdf/2102.05095.pdf), Gedas Bertasius, Heng Wang, Lorenzo Torresani
