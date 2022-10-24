[简体中文](../../../zh-CN/model_zoo/recognition/ctrgcn_light.md) | English

# CTRGCN_light (CTR-GCN's lightweight model series)

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

**We believe that trivial structures are detrimental to GPU parallel acceleration and will bring structural complexity (resulting in excessive file size of inference model structures).** Based on CTRGCN, under the condition that the inference model is less than 10M, we propose the lightweight model series CTRGCN_light, which includes two models: CTRGCN_lightV1 and CTRGCN_lightV2. Compared with CTRGCN, the same lightweight operations made by the entire CTRGCN_light series is: First, extract the small 1x1 convolutions of each branch of the Temporal Modeling module in Figure (a) below, merge them into a large 1x1 convolution, and then allocate the corresponding 1x1 convolution output to each branch at the expense of memory division in forward propagation; Second, num_subset CTR-GC blocks of the Spatial Modeling module in Figure (a) below are merged into a CTR-GC block, and the 1x1 convolutions used for feature mapping and having the same input in the original CTR-GC in Figure (b) are merged together, that is, module fusion and convolutional fusion are performed to reduce trivial structures, but both are at the expense of memory division in forward propagation.

<div align="center">
<img src="../../../images/ctrgcn_light.png" height=150 width=650 hspace='10'/> <br />
</div>

## Data

Please refer to NTU-RGBD data download and preparation doc [NTU-RGBD](../../dataset/ntu-rgbd.md)


## Train


### Train on NTU-RGBD

- Train CTRGCN_light on NTU-RGBD scripts using single gpu：

```bash
# joint modality for CTRGCN_lightV1
python main.py --validate -c configs/recognition/ctrgcn/ctrgcn_light_v1_ntucs_joint.yaml --seed 1

# joint modality for CTRGCN_lightV2
python main.py --validate -c configs/recognition/ctrgcn/ctrgcn_light_v2_ntucs_joint.yaml --seed 1
```

- Train CTRGCN_light on NTU-RGBD scriptsusing multi gpus:

```bash
# for CTRGCN_lightV1
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_ctrgcn_light_v1  main.py  --validate -c configs/recognition/ctrgcn/ctrgcn_light_v1_ntucs_joint.yaml

# for CTRGCN_lightV2
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_ctrgcn_light_v2  main.py  --validate -c configs/recognition/ctrgcn/ctrgcn_light_v2_ntucs_joint.yaml
```

- config file `ctrgcn_light_v1_ntucs_joint.yaml` corresponding to the config of CTRGCN_light on NTU-RGB+D dataset with cross-subject splits.

## Test

### Test on NTU-RGB+D

- Test scripts：

```bash
# joint modality for CTRGCN_lightV1
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_light_v1_ntucs_joint.yaml -w data/CTRGCN_lightV1_ntucs_joint.pdparams

# joint modality for CTRGCN_lightV2
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_light_v2_ntucs_joint.yaml -w data/CTRGCN_lightV2_ntucs_joint.pdparams
```

- Specify the config file with `-c`, specify the weight path with `-w`.

Under the joint modality, X-sub evaluation standard, the experimental accuracy (for the CTRGCN_light series is averaged 3 times) and the inference speed (measured on the AIStudio Supreme Card) on the NTU-RGB+D dataset are as follows:
| Model | Top-1(%) | the size of inference model |GPU inference speed (ms) (lower is better)|CPU inference speed (ms) (lower is better)| checkpoints|
| -------- | -------- | -------- |-------- | -------- |-------- |
| CTRGCN  | 89.93     | 约14.5MB     | 31.4189 | 302.6343 |[CTRGCN_ntucs_joint.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/CTRGCN_ntucs_joint.pdparams) |
| CTRGCN_lightV1 | 91.126(±0.272) | 约9.6MB | 15.3605 | 238.6627 |[CTRGCN_lighjtV1_ntucs_joint.pdparams](https://aistudio.baidu.com/aistudio/datasetdetail/173887) |
| CTRGCN_lightV2 | 91.042(±0.153) | 约9.8MB | 18.233 | 177.4101 |[CTRGCN_lighjtV2_ntucs_joint.pdparams](https://aistudio.baidu.com/aistudio/datasetdetail/173887)|
| CTRGCN_lightV2_dml | 91.187 | 约9.8MB | 18.233 | 177.4101 |[CTRGCN_lighjtV2_ntucs_joint_dml.pdparams](https://aistudio.baidu.com/aistudio/datasetdetail/173887)|


## Inference

### export inference model

```bash
# for CTRGCN_lightV1
python3.7 tools/export_model.py -c configs/recognition/ctrgcn/ctrgcn_light_v1_ntucs_joint.yaml \
                                -p data/CTRGCN_lightV1_ntucs_joint.pdparams \
                                -o inference/CTRGCN_lightV1
                                
# for CTRGCN_lightV2
python3.7 tools/export_model.py -c configs/recognition/ctrgcn/ctrgcn_light_v2_ntucs_joint.yaml \
                                -p data/CTRGCN_lightV2_ntucs_joint.pdparams \
                                -o inference/CTRGCN_lightV2
```

 To get model architecture file`CTRGCN_lightV2_joint.pdmodel`and parameters file`CTRGCN_lightV2_joint.pdiparams`, use:

- Args usage please refer to [Model Inference](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86).

### infer

```bash
# for CTRGCN_lightV2
python3.7 tools/predict.py --input_file data/example_NTU-RGB-D_sketeton.npy \
                           --config configs/recognition/ctrgcn/ctrgcn_light_v2_ntucs_joint.yaml \
                           --model_file inference/CTRGCN_lightV2/CTRGCN_lightV2_joint.pdmodel \
                           --params_file inference/CTRGCN_lightV2/CTRGCN_lightV2_joint.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

example of logs:

```
Current video file: data/example_NTU-RGB-D_sketeton.npy
        top-1 class: 58
        top-1 score: 0.6479401
```


## Reference

- [Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition](https://arxiv.org/abs/2107.12213), Chen, Yuxin and Zhang, Ziqi and Yuan, Chunfeng and Li, Bing and Deng, Ying and Hu, Weiming

