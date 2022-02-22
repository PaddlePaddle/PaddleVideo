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
python main.py --validate -c configs/recognition/ctrgcn/ctrgcn_ntucs.yaml --seed 1
```

- Train CTR-GCN on NTU-RGBD scriptsusing multi gpus:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_ctrgcn  main.py  --validate -c configs/recognition/ctrgcn/ctrgcn_ntucs.yaml
```

- config file `ctrgcn_ntucs.yaml` corresponding to the config of CTR-GCN on NTU-RGB+D dataset with cross-subject splits.


## Test

### Test on NTU-RGB+D

- Test scripts：

```bash
python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_ntucs.yaml -w output/CTRGCN/CTRGCN_best.pdparams
```

- Specify the config file with `-c`, specify the weight path with `-w`.


Accuracy on NTU-RGB+D dataset:

| split | Top-1 | checkpoints |
| :----: | :----: | :---- |
| cross-subject | 86.02 | [CTRGCN_ntucs.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/CTRGCN_ntucs.pdparams) |


## Inference

### export inference model

```bash
python3.7 tools/export_model.py -c configs/recognition/ctrgcn/ctrgcn_ntucs.yaml \
                                -p data/CTRGCN_ntucs.pdparams \
                                -o inference/STGCN
```

 To get model architecture file `CTRGCN.pdmodel` and parameters file `CTRGCN.pdiparams`, use:

- Args usage please refer to [Model Inference](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86).

### infer

```bash
python3.7 tools/predict.py --input_file data/example_NTU-RGB-D_sketeton.npy \
                           --config configs/recognition/ctrgcn/ctrgcn_ntucs.yaml \
                           --model_file inference/CTRGCN/CTRGCN.pdmodel \
                           --params_file inference/CTRGCN/CTRGCN.pdiparams \
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
