[简体中文](../../../zh-CN/model_zoo/recognition/movinet.md) | English

# MoViNet

---
## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Reference](#Reference)

## Introduction

Movinet is a mobile video network developed by Google research. It uses causal convolution operator with stream buffer and temporal ensembles to improve accuracy. It is a lightweight and efficient video model that can be used for online reasoning video stream.


## Data

Please refer to Kinetics400 data download and preparation doc [k400-data](../../dataset/K400.md)


## Train

- Train MoViNet on kinetics-400 scripts:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_movinet main.py --validate -c configs/recognition/movinet/movinet_k400_frame.yaml
```

## Test

- For uniform sampling, test accuracy can be found in training-logs by search key word `best`, such as:

```txt
Already save the best model (top1 acc)0.6489
```

- Test scripts:

```bash
python3.7 main.py --test -c configs/recognition/movinet/movinet_k400_frame.yaml -w output/MoViNet/MoViNet_best.pdparams
```


Accuracy on Kinetics400:

| Config | Sampling method | num_seg | target_size | Top-1 | checkpoints |
| :------: | :--------: | :-------: | :-------: | :-----: | :-----: |
| A0 | Uniform | 50 | 172  | 66.62 | [MoViNetA0_k400.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/MoViNetA0_k400.pdparams)  |

## Inference

### export inference model

 To get model architecture file `MoViNetA0.pdmodel` and parameters file `MoViNetA0.pdiparams`, use:

```bash
python3.7 tools/export_model.py -c configs/recognition/movinet/movinet_k400_frame.yaml \
                                -p data/MoViNetA0_k400.pdparams \
                                -o inference/MoViNetA0
```

- Args usage please refer to [Model Inference](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86).

### infer

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/movinet/movinet_k400_frame.yaml \
                           --model_file inference/MoViNetA0/MoViNet.pdmodel \
                           --params_file inference/MoViNetA0/MoViNet.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

example of logs:

```
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 0.7667049765586853
```

## Reference

- [MoViNets: Mobile Video Networks for Efficient Video Recognition](https://arxiv.org/abs/2103.11511)
