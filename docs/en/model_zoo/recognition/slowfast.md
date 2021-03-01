[简体中文 ](../../../zh-CN/model_zoo/recognition/slowfast.md) | English

# SlowFast

---
## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Reference](#Reference)


## Introduction

SlowFast  involves (i) a Slow pathway, operating at low frame rate, to capture spatial semantics, and (ii) a Fast path-way, operating at high frame rate, to capture motion at fine temporal resolution. The Fast pathway can be made very lightweight by reducing its channel capacity, yet can learn useful temporal information for video recognition.

<p align="center">
<img src="../../../images/SlowFast.png" height=300 width=500 hspace='10'/> <br />
SlowFast Overview
</p>


## Data

We use Kinetics-400 to train this model，data preparation please refer to [Kinetics-400 dataset](../../dataset/k400.md).


## Train

You can start training by：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_slowfast  main.py --validate -c configs/recognition/slowfast/slowfast.yaml 
```

- Training would be efficent using our code. The training speed is 2x faster than the original implementation. Details can refer to [benchmark](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/en/benchmark.md).


## Test

You can start testing by：

```bash
python -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_slowfast_test main.py --test -c  configs/recognition/slowfast/slowfast.yaml -w output/SlowFast/SlowFast_epoch_000196.pdparams
```

-  Args `-w` is used to specifiy the model path，you can download our model in [SlowFast.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast.pdparams).


Test accuracy in Kinetics-400:

| Acc1 | Acc5 |
| :---: | :---: |
| 74.35 | 91.33 |

- Acc1 may be lower than that released in papaer, as ~5% data of kinetics-400 is missing. Experiments have verified that if training with the same data, we can get the same accuracy.

## Reference

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982), Feichtenhofer C, Fan H, Malik J, et al. 
