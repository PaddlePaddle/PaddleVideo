[简体中文](../../../zh-CN/model_zoo/recognition/posec3d.md) | English

# PoseC3D

---
## Contents

- [PoseC3D](#PoseC3D)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Data](#data)
  - [Train](#train)
    - [Train on UCF101.](#train-on-ucf101)
  - [Test](#test)
    - [Test onf UCF101](#test-onf-ucf101)
  - [Inference](#inference)
    - [export inference model](#export-inference-model)
    - [infer](#infer)
  - [Reference](#reference)


## Introduction

Human  skeleton,  as  a  compact  representation  of  hu-man  action,  has  received  increasing  attention  in  recentyears.    Many  skeleton-based  action  recognition  methodsadopt graph convolutional networks (GCN) to extract fea-tures on top of human skeletons.   Despite the positive re-sults  shown  in  previous  works,  GCN-based  methods  aresubject  to  limitations  in  robustness,  interoperability,  andscalability.  In this work, we propose PoseC3D, a new ap-proach  to  skeleton-based  action  recognition,  which  relieson  a  3D  heatmap  stack  instead  of  a  graph  sequence  asthe base representation of human skeletons.  Compared toGCN-based methods, PoseC3D is more effective in learningspatiotemporal features, more robust against pose estima-tion noises, and generalizes better in cross-dataset settings.Also, PoseC3D can handle multiple-person scenarios with-out additional computation cost, and its features can be eas-ily integrated with other modalities at early fusion stages,which  provides  a  great  design  space  to  further  boost  theperformance. On four challenging datasets, PoseC3D con-sistently obtains superior performance, when used alone onskeletons and in combination with the RGB modality.

## Data

Please download UCF101 skeletons datasets and pretraind model weights.

[https://aistudio.baidu.com/aistudio/datasetdetail/140593](https://aistudio.baidu.com/aistudio/datasetdetail/140593)

## Train

### Train on UCF101.

- Train PoseC3D model:

```bash
python3.7 main.py --validate -c configs/recognition/posec3d/posec3d.yaml --weights res3d_k400.pdparams
```


## Test

### Test onf UCF101

- Test scripts：

```bash
python3.7 main.py --test -c configs/recognition/posec3d/posec3d.yaml  -w output/PoseC3D/PoseC3D_epoch_0012.pdparams
```

- Specify the config file with `-c`, specify the weight path with `-w`.


Accuracy on UCF101 dataset:

| Test_Data | Top-1 | checkpoints |
| :----: | :----: | :---- |
| UCF101 test1 | 87.05 | [PoseC3D_ucf101.pdparams]() |



## Inference

### export inference model

 To get model architecture file `PoseC3D.pdmodel` and parameters file `PoseC3D.pdiparams`, use:

```bash
python3.7 tools/export_model.py -c configs/recognition/posec3d/posec3d.yaml \
                                -p data/PoseC3D_ucf101.pdparams \
                                -o inference/PoseC3D
```

- Args usage please refer to [Model Inference](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86).

### infer

```bash
python3.7 tools/predict.py --input_file data/example_UCF101_skeleton.pkl\
                           --config configs/recognition/posec3d/posec3d.yaml \
                           --model_file inference/PoseC3D/PoseC3D.pdmodel \
                           --params_file inference/PoseC3D/PoseC3D.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

example of logs:

```
Current video file: data/example_UCF101_skeleton.pkl
	top-1 class: 0
	top-1 score: 0.6731489896774292
```


## Reference

- [Revisiting Skeleton-based Action Recognition](https://arxiv.org/pdf/2104.13586v1.pdf), Haodong Duan, Yue Zhao, Kai Chen, Dian Shao, Dahua Lin, Bo Dai
