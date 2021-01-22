[简体中文 ](../../../zh-CN/model_zoo/localization/bmn.md) | English

# BMN

---
## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Reference](#Reference)


## Introduction

BMN model contains three modules: Base Module handles the input feature sequence, and out- puts feature sequence shared by the following two modules; Temporal Evaluation Module evaluates starting and ending probabilities of each location in video to generate boundary probability sequences; Proposal Evaluation Module con- tains the BM layer to transfer feature sequence to BM fea- ture map, and contains a series of 3D and 2D convolutional layers to generate BM confidence map.

<p align="center">
<img src="https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/images/BMN.png" height=300 width=400 hspace='10'/> <br />
BMN Overview
</p>


## Data

We use ActivityNet dataset to train this model，data preparation please refer to [ActivityNet dataset](../../dataset/ActivityNet.md).


## Train

You can start training by such command：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -B -m paddle.distributed.launch --gpus="0,1,2,3"  --log_dir=log_bmn main.py  --validate -c configs/localization/bmn.yaml
```


## Test

You can start testing by such command：

```bash
python main.py --test -c configs/localization/bmn.yaml -w output/BMN/BMN_epoch_00010.pdparams -o DATASET.batch_size=1
```

-  Args `-w` is used to specifiy the model path，you can download our model in [BMN.pdparams](https://videotag.bj.bcebos.com/PaddleVideo/BMN/BMN.pdparams)

Test accuracy in Kinetics-400:

| Acc1 | Acc5 |
| :---: | :---: |
| 74.35 | 91.33 |

- Acc1 may be lower than that released in papaer, as ~5% data of kinetics-400 is missing. Experiments have verified that if training with the same data, we can get the same accuracy.

## Reference

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982), Feichtenhofer C, Fan H, Malik J, et al. 
