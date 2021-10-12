[简体中文](../../../zh-CN/model_zoo/recognition/agcn.md) | English

# AGCN

---
## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Reference](#Reference)


## Introduction

We implemented Adaptive Graph Convolution Network to improve the accuracy of [ST-GCN](./stgcn.md).

## Data

Please refer to FSD-10 data download and preparation doc [FSD-10](../../dataset/fsd10.md)

Please refer to NTU-RGBD data download and preparation doc [NTU-RGBD](../../dataset/ntu-rgbd.md)

## Train

### Train on FSD-10

- Train AGCN on FSD-10 scripts:

```bash
python3.7 main.py -c configs/recognition/agcn/agcn_fsd.yaml
```

- Turn off `valid` when training, as validation dataset is not available for the competition.

### Train on NTU-RGBD

- Train AGCN on NTU-RGBD scripts:

```bash
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_agcn  main.py  --validate -c configs/recognition/agcn/agcn_ntucs.yaml
```

- config file `agcn_ntucs.yaml` corresponding to the config of AGCN on NTU-RGB+D dataset with cross-subject splits.


## Test

### Test onf FSD-10

- Test scripts：

```bash
python3.7 main.py --test -c configs/recognition/agcn/agcn_fsd.yaml  -w output/AGCN/AGCN_epoch_00100.pdparams
```

- Specify the config file with `-c`, specify the weight path with `-w`.

- Evaluation results will be saved in `submission.csv` file, final score can be obtained in [competition website](https://aistudio.baidu.com/aistudio/competition/detail/115).

Accuracy on FSD-10 dataset:

| Test_Data | Top-1 | checkpoints |
| :----: | :----: | :---- |
| Test_A | 90.66 | AGCN_fsd.pdparams|


### Test on NTU-RGB+D

- Test scripts：

```bash
python3.7 main.py --test -c configs/recognition/agcn/agcn_ntucs.yaml -w output/AGCN/AGCN_best.pdparams
```

- Specify the config file with `-c`, specify the weight path with `-w`.

Accuracy on NTU-RGB+D dataset:

| split | Top-1 | checkpoints |
| :----: | :----: | :---- |
| cross-subject | 83.27 | AGCN_ntucs.pdparams|


## Inference

### export inference model

 To get model architecture file `AGCN.pdmodel` and parameters file `AGCN.pdiparams`, use:

```bash
python3.7 tools/export_model.py -c configs/recognition/agcn/agcn_fsd.yaml \
                                -p data/AGCN_fsd.pdparams \
                                -o inference/AGCN
```

- Args usage please refer to [Model Inference](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86).

### infer

```bash
python3.7 tools/predict.py --input_file data/fsd10/example_skeleton.npy \
                           --config configs/recognition/agcn/agcn_fsd.yaml \
                           --model_file inference/AGCN/AGCN.pdmodel \
                           --params_file inference/AGCN/AGCN.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

example of logs:

```
Current video file: data/fsd10/example_skeleton.npy
        top-1 class: 0
        top-1 score: 0.8932635188102722
```


## Reference

- [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455), Sijie Yan, Yuanjun Xiong, Dahua Lin

- [Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1805.07694), Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu

- [Skeleton-Based Action Recognition with Multi-Stream Adaptive Graph Convolutional Networks](https://arxiv.org/abs/1912.06971), Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu

- Many thanks to [li7819559](https://github.com/li7819559) and [ZhaoJingjing713](https://github.com/ZhaoJingjing713) for contributing the code.
