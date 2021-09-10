[简体中文](../../../zh-CN/model_zoo/recognition/pp-tsm.md) | English

# PP-TSM

---
## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Reference](#Reference)

## Introduction

We optimized TSM model and proposed **PP-TSM** in this repo. Without increasing the number of parameters, the accuracy of TSM was significantly improved in UCF101 and Kinetics-400 datasets. Please refer to [**Tricks on PP-TSM**](https://zhuanlan.zhihu.com/p/382134297) for more details.

| Version | Sampling method | Top1 |
| :------ | :----------: | :----: |
| Ours (distill) | Dense | **76.16** |
| Ours | Dense | 75.69 |
| [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README.md) | Dense | 74.55 |
| [mit-han-lab](https://github.com/mit-han-lab/temporal-shift-module) | Dense | 74.1 |


| Version | Sampling method | Top1 |
| :------ | :----------: | :----: |
| Ours (distill) | Uniform | **75.11** |
| Ours | Uniform | 74.54 |
| [mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README.md) |  Uniform | 71.90 |
| [mit-han-lab](https://github.com/mit-han-lab/temporal-shift-module)  | Uniform | 71.16 |


## Data

Please refer to Kinetics400 data download and preparation doc [k400-data](../../dataset/K400.md)

Please refer to UCF101 data download and preparation doc [ucf101-data](../../dataset/ucf101.md)


## Train

### Train on kinetics-400

#### download pretrain-model

Please download [ResNet50_vd_ssld_v2](https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams) as pretraind model:

```bash
wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams
```

and add path to `MODEL.framework.backbone.pretrained` in config file as：

```yaml
MODEL:
    framework: "Recognizer2D"
    backbone:
        name: "ResNetTweaksTSM"
        pretrained: your weight path
```

#### Start training

- Train PP-TSM on kinetics-400 scripts:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml
```

- Train PP-TSM on kinetics-400 video data using scripts:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/pptsm/pptsm_k400_videos_uniform.yaml
```

- AMP is useful for speeding up training:

```bash
export FLAGS_conv_workspace_size_limit=800 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --amp --validate -c configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml
```

- Train PP-TSM on kinetics-400 with dense sampling:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/pptsm/pptsm_k400_frames_dense.yaml
```


## Test

- For uniform sampling, test accuracy can be found in training-logs by search key word `best`, such as:

```txt
Already save the best model (top1 acc)0.7454
```

- For dense sampling, test accuracy can be obtained using scripts:

```bash
python3 main.py --test -c configs/recognition/pptsm/pptsm_k400_frames_dense.yaml -w output/ppTSM/ppTSM_best.pdparams
```


Accuracy on Kinetics400:

| backbone | distill | Sampling method | num_seg | target_size | Top-1 | checkpoints |
| :------: | :----------: | :----: | :----: | :----: | :----: | :---- |
| ResNet50 | False | Uniform | 8 | 224 | 74.54 | [ppTSM_k400_uniform.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_uniform.pdparams) |
| ResNet50 | False | Dense | 8 | 224 | 75.69 | [ppTSM_k400_dense.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense.pdparams) |
| ResNet50 | True | Uniform | 8 | 224 | 75.11 | [ppTSM_k400_uniform_distill.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_uniform_distill.pdparams) |
| ResNet50 | True | Dense | 8 | 224 | 76.16 | [ppTSM_k400_dense_distill.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.1/PPTSM/ppTSM_k400_dense_distill.pdparams) |


## Inference

### export inference model

 To get model architecture file `ppTSM.pdmodel` and parameters file `ppTSM.pdiparams`, use:

```bash
python3.7 tools/export_model.py -c configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml \
                                -p data/ppTSM_k400_uniform.pdparams \
                                -o inference/ppTSM
```

- Args usage please refer to [Model Inference](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86).

### infer

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml \
                           --model_file inference/ppTSM/ppTSM.pdmodel \
                           --params_file inference/ppTSM/ppTSM.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

example of logs:

```
Current video file: data/example.avi
	top-1 class: 5
	top-1 score: 0.9907386302947998
```

we can get the class name using class id and map file `data/k400/Kinetics-400_label_list.txt`. The top1 prediction of `data/example.avi` is `archery`.
- **Note**: For models that combine N and T during calculation (such as TSN, TSM), when `use_tensorrt=True`, you need to specify the `batch_size` argument as batch_size*num_seg.

    ```bash
    python3.7 tools/predict.py --input_file data/example.avi \
                               --config configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml \
                               --model_file inference/ppTSM/ppTSM.pdmodel \
                               --params_file inference/ppTSM/ppTSM.pdiparams \
                               --batch_size 8 \
                               --use_gpu=True \
                               --use_tensorrt=True
    ```
## Reference

- [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/pdf/1811.08383.pdf), Ji Lin, Chuang Gan, Song Han
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), Geoffrey Hinton, Oriol Vinyals, Jeff Dean
