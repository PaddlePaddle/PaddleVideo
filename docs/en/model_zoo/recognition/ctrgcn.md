[简体中文](../../../zh-CN/model_zoo/recognition/ctrgcn.md) | English

# CTR-GCN

---
## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Reference](#Reference)


## Introduction

[CTRGCN](https://github.com/Uason-Chen/CTR-GCN.git) is a bone based behavior recognition model proposed by iccv 2021. By applying the changes to the graph convolution of human bone data with topological structure, and using spatio-temporal graph convolution to extract spatio-temporal features for behavior recognition, the accuracy of bone based behavior recognition task is greatly improved.

<div align="center">
<img src="../../../images/ctrgcn.jpg" height=200 width=950 hspace='10'/> <br />
</div>


## Data

Please refer to NTU-RGBD data download and preparation doc [NTU-RGBD](../../dataset/ntu-rgbd.md)


## Train


### Train on NTU-RGBD

- Train CTR-GCN on NTU-RGBD scripts using single gpu：

```bash
# joint modality
python main.py --validate -c configs/recognition/ctrgcn/ctrgcn_ntucs_joint.yaml --seed 1

# bone modality
python main.py --validate -c configs/recognition/ctrgcn/ctrgcn_ntucs_bone.yaml --seed 1

# motion modality
python main.py --validate -c configs/recognition/ctrgcn/ctrgcn_ntucs_motion.yaml --seed 1

# bone motion modality
python main.py --validate -c configs/recognition/ctrgcn/ctrgcn_ntucs_bone_motion.yaml --seed 1
```

- Train CTR-GCN on NTU-RGBD scriptsusing multi gpus:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_ctrgcn  main.py  --validate -c configs/recognition/ctrgcn/ctrgcn_ntucs_joint.yaml
```

- config file `ctrgcn_ntucs_joint.yaml` corresponding to the config of CTR-GCN on NTU-RGB+D dataset with cross-subject splits.


## Test

### Test on NTU-RGB+D

- Test scripts：

```bash
# joint modality
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_ntucs_joint.yaml -w data/CTRGCN_ntucs_joint.pdparams

# bone modality
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_ntucs_bone.yaml -w data/CTRGCN_ntucs_bone.pdparams

# motion modality
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_ntucs_motion.yaml -w data/CTRGCN_ntucs_motion.pdparams

# bone motion modality
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_ntucs_bone_motion.yaml -w data/CTRGCN_ntucs_bone_motion.pdparams
```

- Specify the config file with `-c`, specify the weight path with `-w`.


Accuracy on NTU-RGB+D dataset:

| split | modality | Top-1 | checkpoints |
| :----: | :----: | :----: | :----: |
| cross-subject | joint | 89.93 | [CTRGCN_ntucs_joint.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/CTRGCN_ntucs_joint.pdparams) |
| cross-subject | bone | 85.24 | [CTRGCN_ntucs_bone.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/CTRGCN_ntucs_bone.pdparams) |
| cross-subject | motion | 85.33 | [CTRGCN_ntucs_motion.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/CTRGCN_ntucs_motion.pdparams) |
| cross-subject | bone motion | 84.53 | [CTRGCN_ntucs_bone_motion.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/CTRGCN_ntucs_bone_motion.pdparams) |


## Inference

### export inference model

```bash
python3.7 tools/export_model.py -c configs/recognition/ctrgcn/ctrgcn_ntucs_joint.yaml \
                                -p data/CTRGCN_ntucs_joint.pdparams \
                                -o inference/CTRGCN
```

 To get model architecture file `CTRGCN.pdmodel` and parameters file `CTRGCN.pdiparams`, use:

- Args usage please refer to [Model Inference](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86).

### infer

```bash
python3.7 tools/predict.py --input_file data/example_NTU-RGB-D_sketeton.npy \
                           --config configs/recognition/ctrgcn/ctrgcn_ntucs_joint.yaml \
                           --model_file inference/CTRGCN_joint/CTRGCN_joint.pdmodel \
                           --params_file inference/CTRGCN_joint/CTRGCN_joint.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

example of logs:

```
Current video file: data/example_NTU-RGB-D_sketeton.npy
        top-1 class: 4
        top-1 score: 0.999988317489624
```

## Reference

- [Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition](https://arxiv.org/abs/2107.12213), Chen, Yuxin and Zhang, Ziqi and Yuan, Chunfeng and Li, Bing and Deng, Ying and Hu, Weiming
