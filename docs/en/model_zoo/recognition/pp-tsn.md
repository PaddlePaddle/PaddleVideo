[简体中文](../../../zh-CN/model_zoo/recognition/pp-tsn.md) | English

# PP-TSN

## Content

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Reference](#Reference)


## Introduction

We have improved the [TSN model](./tsn.md) and obtained a more accurate 2D practical video classification model **PP-TSN**. Without increasing the amount of parameters and calculations, the accuracy on the UCF-101, Kinetics-400 and other data sets significantly exceeds the original version. The accuracy on the Kinetics-400 data set is shown in the following table.

| Version | Top1 |
| :------ | :----: |
| Ours (distill) | 75.06 |
| Ours | **73.68** |
| [mmaction2](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn#kinetics-400) | 71.80 |


## Data

K400 data download and preparation please refer to [Kinetics-400 data preparation](../../dataset/k400.md)

UCF101 data download and preparation please refer to [UCF-101 data preparation](../../dataset/ucf101.md)


## Train

### Kinetics-400 data set training

#### Download and add pre-trained models

1. Download the image distillation pre-training model [ResNet50_vd_ssld_v2.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams) as the Backbone initialization parameter, or download it through wget

   ```bash
   wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams
   ```

2. Open `PaddleVideo/configs/recognition/pptsn/pptsn_k400_frames.yaml`, and fill in the downloaded weight storage path below `pretrained:`

    ```yaml
    MODEL:
        framework: "Recognizer2D"
        backbone:
            name: "ResNetTweaksTSN"
            pretrained: fill in the path here
    ```

#### Start training

- The Kinetics400 data set uses 8 cards for training, and the start command of the training method is as follows:

    ```bash
    # frames data format
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_pptsn main.py --validate -c configs/recognition/ pptsn/pptsn_k400_frames.yaml

    # videos data format
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_pptsn main.py --validate -c configs/recognition/ pptsn/pptsn_k400_videos.yaml
    ```

- Turn on amp mixed-precision training to speed up the training process. The training start command is as follows:

    ```bash
    export FLAGS_conv_workspace_size_limit=800 # MB
    export FLAGS_cudnn_exhaustive_search=1
    export FLAGS_cudnn_batchnorm_spatial_persistent=1

    # frames data format
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_pptsn main.py --amp --validate -c configs /recognition/pptsn/pptsn_k400_frames.yaml

    # videos data format
    python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_pptsn main.py --amp --validate -c configs /recognition/pptsn/pptsn_k400_videos.yaml
    ```

- In addition, you can customize and modify the parameter configuration to achieve the purpose of training/testing on different data sets. It is recommended that the naming method of the configuration file is `model_dataset name_file format_data format_sampling method.yaml` , Please refer to [config](../../tutorials/config.md) for parameter usage.


## Test

- The PP-TSN model is verified during training. You can find the keyword `best` in the training log to obtain the model test accuracy. The log example is as follows:

	```
  Already save the best model (top1 acc)0.7004
	```

- Since the sampling method of the PP-TSN model test mode is **TenCrop**, which is slightly slower but more accurate, it is different from the **CenterCrop** used in the verification mode during the training process, so the verification index recorded in the training log is `topk Acc `Does not represent the final test score, so after the training is completed, you can use the test mode to test the best model to obtain the final index, the command is as follows:

	```bash
  python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_pptsn main.py --test -c configs/recognition/ pptsn/pptsn_k400_frames.yaml -w "output/ppTSN/ppTSN_best.pdparams"
	```

	When the test configuration uses the following parameters, the test indicators on the validation data set of Kinetics-400 are as follows:

	| backbone | Sampling method | distill | num_seg | target_size | Top-1 |       checkpoints       |
	| :------: | :-------------: | :-----: | :-----: | :---------: | :---- | :---------------------: |
	| ResNet50 |     TenCrop     |  False  |    3    |     224     | 73.68 | [ppTSN_k400.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSN_k400.pdparams) |
	| ResNet50 |     TenCrop     |  True   |    8    |     224     | 75.06 | [ppTSN_k400_8.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/ppTSN_k400_8.pdparams) |

- The PP-TSN video sampling strategy is TenCrop sampling: in time sequence, the input video is evenly divided into num_seg segments, and the middle position of each segment is sampled 1 frame; spatially, from the upper left corner, upper right corner, center point, lower left corner, and lower right corner Each of the 5 sub-regions sampled an area of 224x224, and the horizontal flip was added to obtain a total of 10 sampling results. A total of 1 clip is sampled for 1 video.

- Distill is `True`, which means that the pre-trained model obtained by distillation is used. For the specific distillation scheme, please refer to [ppTSM Distillation Scheme]().


## Inference

### Export inference model

```bash
python3.7 tools/export_model.py -c configs/recognition/pptsn/pptsn_k400_frames.yaml -p data/ppTSN_k400.pdparams -o inference/ppTSN
```

The above command will generate the model structure file `ppTSN.pdmodel` and model weight files `ppTSN.pdiparams` and `ppTSN.pdiparams.info` files required for prediction, all of which are stored in the `inference/ppTSN/` directory

For the meaning of each parameter in the above bash command, please refer to [Model Reasoning Method](https://github.com/HydrogenSulfate/PaddleVideo/blob/PPTSN-v1/docs/en/start.md#2-infer)

### Use prediction engine inference

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/pptsn/pptsn_k400_frames.yaml \
                           --model_file inference/ppTSN/ppTSN.pdmodel \
                           --params_file inference/ppTSN/ppTSN.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

The output example is as follows:

```bash
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 0.998979389667511
```

It can be seen that using the PP-TSN model trained on Kinetics-400 to predict `data/example.avi`, the output top1 category id is `5`, and the confidence is 0.99. By consulting the category id and name correspondence table `data/k400/Kinetics-400_label_list.txt`, it can be known that the predicted category name is `archery`.
- **Note**: For models that combine N and T during calculation (such as TSN, TSM), when `use_tensorrt=True`, you need to specify the `batch_size` argument as batch_size\*num_seg.

    ```bash
    python3.7 tools/predict.py --input_file data/example.avi \
                               --config configs/recognition/pptsn/pptsn_k400_frames.yaml \
                               --model_file inference/ppTSN/ppTSN.pdmodel \
                               --params_file inference/ppTSN/ppTSN.pdiparams \
                               --batch_size 8 \
                               --use_gpu=True \
                               --use_tensorrt=True
    ```
## Reference

- [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/pdf/1608.00859.pdf), Limin Wang, Yuanjun Xiong, Zhe Wang
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), Geoffrey Hinton, Oriol Vinyals, Jeff Dean
