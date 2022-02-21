[简体中文](../../../zh-CN/model_zoo/segmentation/mstcn.md) | English

# MS-TCN : Video Action Segmentation Model

---
## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Train](#Train)
- [Test](#Test)
- [Inference](#Inference)
- [Reference](#Reference)

## Introduction

Ms-tcn model is a classic model of video motion segmentation model, which was published on CVPR in 2019. We optimized the officially implemented pytorch code and obtained higher precision results in paddlevideo.

<p align="center">
<img src="../../../images/mstcn.PNG" height=300 width=400 hspace='10'/> <br />
MS-TCN Overview
</p>

## Data

MS-TCN can choose 50salads, breakfast, gtea as trianing set. Please refer to Video Action Segmentation dataset download and preparation doc [Video Action Segmentation dataset](../../dataset/SegmentationDataset.md)

## Train

After prepare dataset, we can run sprits.

```bash
# gtea dataset
export CUDA_VISIBLE_DEVICES=3
python3.7 main.py  --validate -c configs/segmentation/ms_tcn/ms_tcn_gtea.yaml --seed 1538574472
```

- Start the training by using the above command line or script program. There is no need to use the pre training model. The video action segmentation model is usually a full convolution network. Due to the different lengths of videos, the `DATASET.batch_size` of the video action segmentation model is usually set to `1`, that is, batch training is not required. At present, only **single sample** training is supported.

## Test

Test MS-TCN on dataset scripts:

```bash
python main.py  --test -c configs/segmentation/ms_tcn/ms_tcn_gtea.yaml --weights=./output/MSTCN/MSTCN_split_1_best.pdparams
```

- The specific implementation of the index is to calculate ACC, edit and F1 scores by referring to the test script[evel.py](https://github.com/yabufarha/ms-tcn/blob/master/eval.py) provided by the author of ms-tcn.

- The evaluation method of data set adopts the folding verification method in ms-tcn paper, and the division method of folding is the same as that in ms-tcn paper.

Accuracy on Breakfast dataset(4 folding verification):

| Model | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| paper | 66.3% | 61.7% | 48.1% | 48.1% | 37.9% |
| paddle | 65.2% | 61.5% | 53.7% | 49.2% | 38.8% |

Accuracy on 50salads dataset(5 folding verification):

| Model | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| paper | 80.7% | 67.9% | 76.3% | 74.0% | 64.5% |
| paddle | 81.1% | 71.5% | 77.9% | 75.5% | 66.5% |

Accuracy on gtea dataset(4 folding verification):

| Model | Acc | Edit | F1@0.1 | F1@0.25 | F1@0.5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| paper | 79.2% | 81.4% | 87.5% | 85.4% | 74.6% |
| paddle | 76.9% | 81.8% | 86.4% | 84.7% | 74.8% |

Model weight for gtea

Test_Data| F1@0.5 | checkpoints |
| :----: | :----: | :---- |
| gtea_split1 | 70.2509 | [MSTCN_split_1_best.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/MSTCN_split_1_best.pdparams) |
| gtea_split2 | 70.7224 | [MSTCN_split_2_best.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/MSTCN_split_2_best.pdparams) |
| gtea_split3 | 80.0 | [MSTCN_split_3_best.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/MSTCN_split_3_best.pdparams) |
| gtea_split4 | 78.1609 | [MSTCN_split_4_best.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.2/MSTCN_split_4_best.pdparams) |

## Infer

- Args usage please refer to [Model Inference](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86).

## Reference

- [MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://arxiv.org/pdf/1903.01945.pdf), Y. Abu Farha and J. Gall.
