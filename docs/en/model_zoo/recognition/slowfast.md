[简体中文 ](../../../zh-CN/model_zoo/recognition/slowfast.md) | English

# SlowFast

---
## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
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

### Speed up training

It's time consuming to train SlowFast model.  So we implement [Multigrid training stragety](https://arxiv.org/abs/1912.00998) to speed up training. Training script:

```bash
python -B -m paddle.distributed.launch --selected_gpus="0,1,2,3,4,5,6,7" --log_dir=log-slowfast main.py --validate --multigrid -c configs/recognition/slowfast/slowfast_multigrid.yaml
```

Performance evaluation:

| training stragety | time cost of one epoch/min | total training time/min | speed-up |
| :------ | :-----: | :------: |:------: |
| Multigrid | 27.25 |  9758 (6.7 days) | 2.89x |
| Normal | 78.76 | 15438 (10.7days) | base |

For more details, please refer to [accelerate doc](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/tutorials/accelerate.md#%E8%AE%AD%E7%BB%83%E7%AD%96%E7%95%A5%E5%8A%A0%E9%80%9F).


## Test

You can start testing by：

```bash
python -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir=log_slowfast_test main.py --test -c  configs/recognition/slowfast/slowfast.yaml -w output/SlowFast/SlowFast_epoch_000196.pdparams
```

-  Args `-w` is used to specifiy the model path，you can download our model in [SlowFast.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast.pdparams).


Test accuracy in Kinetics-400:

| Configs | Acc1 | Acc5 | Weights |
| :---: | :---: | :---: | :---: |
|  [slowfast.yaml](../../../../configs/recognition/slowfast/slowfast.yaml) | 74.35 | 91.33 | [slowfast_4x16.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast.pdparams) |
|  [slowfast_multigrid.yaml](../../../../configs/recognition/slowfast/slowfast_multigrid.yaml) | 75.84  | 92.33 | [slowfast_8x8.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/SlowFast/SlowFast_8*8.pdparams) |

- Acc1 may be lower than that released in papaer, as ~5% data of kinetics-400 is missing. Experiments have verified that if training with the same data, we can get the same accuracy.


## Inference

### export inference model

 To get model architecture file `SlowFast.pdmodel` and parameters file `SlowFast.pdiparams`, use:

```bash
python3.7 tools/export_model.py -c configs/recognition/slowfast/slowfast.yaml \
                                -p data/SlowFast.pdparams \
                                -o inference/SlowFast
```

- Args usage please refer to [Model Inference](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86).

### infer

```bash
python3.7 tools/predict.py --input_file data/example.avi \
                           --config configs/recognition/slowfast/slowfast.yaml \
                           --model_file inference/SlowFast/SlowFast.pdmodel \
                           --params_file inference/SlowFast/SlowFast.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

example of logs:

```
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 1.0
```

we can get the class name using class id and map file `data/k400/Kinetics-400_label_list.txt`. The top1 prediction of `data/example.avi` is `archery`.


## Reference

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982), Feichtenhofer C, Fan H, Malik J, et al.
