[简体中文](../../../zh-CN/model_zoo/recognition/stgcn_plusplus.md) | English

# STGCN++

---
## Contents

- [STGCN++](#STGCN++)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Data](#data)
  - [Train](#train)
    - [Train on NTURGB+D](#train-on-ntu-rgb)
  - [Test](#test)
    - [Test on NTURGB+D](#test-onf-ntu-rgb)
  - [Inference](#inference)
    - [export inference model](#export-inference-model)
    - [infer](#infer)
  - [Reference](#reference)


## Introduction

Human  skeleton,  as  a  compact  representation  of  hu-man  action,  has  received  increasing  attention  in  recentyears.    Many  skeleton-based  action  recognition  methodsadopt graph convolutional networks (GCN) to extract fea-tures on top of human skeletons.   Despite the positive re-sults  shown  in  previous  works,  GCN-based  methods  aresubject  to  limitations  in  robustness,  interoperability,  andscalability.  We also provide an original GCN-based skeleton action recognition model named ST-GCN++, which achieves competitive recognition performance without any complicated attention schemes, serving as a strong baseline.

## Data

Please download NTURGB+D skeletons datasets.

[https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl)

## Train

### Train on NTURGB+D

- Train STGCN++ model:

```bash
pip install -r requirements.txt
ln -s path/to/ntu60_hrnet.pkl data/ntu60_hrnet.pkl
python -u main.py --validate -c configs/recognition/stgcn_plusplus/stgcn_plusplus_ntucs.yaml
```


## Test

### Test on NTURGB+D

- Test scripts：

```bash
python -u main.py --test -c configs/recognition/stgcn_plusplus/stgcn_plusplus_ntucs.yaml \
                  --weights STGCN_PlusPlus_best.pdparams
```

- Specify the config file with `-c`, specify the weight path with `--weights`.


Accuracy on NTURGB+D dataset:

| Test_Data | Top-1 | checkpoints |
| :----: |:-----:| :---- |
| NTURGB+D | 97.56 | [STGCN_PlusPlus_best.pdparams](https://aistudio.baidu.com/aistudio/datasetdetail/169754) |



## Inference

### export inference model

 To get model architecture file `inference.pdmodel` and parameters file `inference.pdiparams`, use:

```bash
python tools/export_model.py -c configs/recognition/stgcn_plusplus/stgcn_plusplus_ntucs.yaml \
                                 --save_name inference \
                                 -p=STGCN_PlusPlus_best.pdparams \
                                 -o=./output/STGCN_PlusPlus/
```

- Args usage please refer to [Model Inference](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86).

### infer

```bash
python tools/predict.py --config configs/recognition/stgcn_plusplus/stgcn_plusplus_ntucs.yaml \
                        --use_gpu=True \
                        --model_file=./output/STGCN_PlusPlus/inference.pdmodel \
                        --params_file=./output/STGCN_PlusPlus/inference.pdiparams --batch_size=1 \
                        --input_file=./data/stdgcn_plusplus_data/example_ntu60_skeleton.pkl
```

example of logs:

```
Current video file: ./data/stdgcn_plusplus_data/example_ntu60_skeleton.pkl
	top-1 class: 0
	top-1 score: 0.9153057932853699
```


## Reference

- [PYSKL: Towards Good Practices for Skeleton Action Recognition](https://arxiv.org/pdf/2205.09443.pdf), Haodong Duan, Jiaqi Wang, Kai Chen, Dahua Lin
