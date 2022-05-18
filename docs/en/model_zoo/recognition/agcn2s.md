[简体中文](../../../zh-CN/model_zoo/recognition/2sAGCN.md) | English

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

![模型结构图](../../../images/agcn2s.png)

[2s-AGCN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf) is an improved article on ST-GCN published in CVPR2019. It proposes a dual-flow adaptive convolutional network, which improves the shortcomings of the original ST-GCN. In the existing GCN based approach, the topology of the graph is set manually and fixed to all layers and input samples. In addition, the second-order information of bone data (bone length and orientation) is naturally more beneficial and discriminating for motion recognition, which was rarely studied in the methods at that time. Therefore, this paper puts forward a node and bones of two kinds of information fusion based on skeleton shuangliu network, and join in figure convolution adjacency matrix adaptive matrix, a sharp rise in the bones of gesture recognition accuracy, also has laid the foundation for subsequent work (the subsequent basic skeleton gesture recognition are based on the flow of network framework).

## Data

Data download and processing are consistent with CTR-GCN. For details, please refer to [NTU-RGBD Data Preparation](../../dataset/ntu-rgbd.md)

## Train

### Train on NTU-RGBD

Train CTR-GCN on NTU-RGBD scripts using single gpu：

```bash
# train cross subject with bone data
python main.py --validate -c configs/recognition/agcn2s/agcn2s_ntucs_bone.yaml --seed 1
# train cross subject with joint data
python main.py --validate -c configs/recognition/agcn2s/agcn2s_ntucs_joint.yaml --seed 1
# train cross view with bone data
python main.py --validate -c configs/recognition/agcn2s/agcn2s_ntucv_bone.yaml --seed 1
# train cross view with joint data
python main.py --validate -c configs/recognition/agcn2s/agcn2s_ntucv_joint.yaml --seed 1
```

config file `agcn2s_ntucs_joint.yaml` corresponding to the config of 2s-AGCN on NTU-RGB+D dataset with cross-subject splits.

## Test

### Test on NTU-RGB+D

Test scripts：

```bash
# test cross subject with bone data
python main.py --test -c configs/recognition/2sagcn/2sagcn_ntucs_bone.yaml -w data/2SAGCN_ntucs_bone.pdparams
# test cross subject with joint data
python main.py --test -c configs/recognition/2sagcn/2sagcn_ntucs_joint.yaml -w data/2SAGCN_ntucs_joint.pdparams
# test cross view with bone data
python main.py --test -c configs/recognition/2sagcn/2sagcn_ntucv_bone.yaml -w data/2SAGCN_ntucv_bone.pdparams
# test cross view with joint data
python main.py --test -c configs/recognition/2sagcn/2sagcn_ntucv_joint.yaml -w data/2SAGCN_ntucv_joint.pdparams
```

* Specify the config file with `-c`, specify the weight path with `-w`.

Accuracy on NTU-RGB+D dataset:

|                |  CS   |   CV   |
| :------------: | :---: | :----: |
| Js-AGCN(joint) | 85.8% | 94.13% |
| Bs-AGCN(bone)  | 86.7% | 93.9%  |

Train log：[download](https://github.com/ELKYang/2s-AGCN-paddle/tree/main/work_dir/ntu)

VisualDL log：[download](https://github.com/ELKYang/2s-AGCN-paddle/tree/main/runs)

checkpoints：

|                            CS-Js                             |                            CS-Bs                             |                            CV-Js                             |                            CV-Bs                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [ntu_cs_agcn_joint](https://github.com/ELKYang/2s-AGCN-paddle/blob/main/weights/ntu_cs_agcn_joint-48-30674.pdparams) | [ntu_cs_agcn_bone](https://github.com/ELKYang/2s-AGCN-paddle/blob/main/weights/ntu_cs_agcn_bone-44-28170.pdparams) | [ntu_cv_agcn_joint](https://github.com/ELKYang/2s-AGCN-paddle/blob/main/weights/ntu_cv_agcn_joint-38-22932.pdparams) | [ntu_cv_agcn_bone](https://github.com/ELKYang/2s-AGCN-paddle/blob/main/weights/ntu_cv_agcn_bone-49-29400.pdparams) |

## Inference

### export inference model

```bash
python3.7 tools/export_model.py -c configs/recognition/agcn2s/2sagcn_ntucs_joint.yaml \
                                -p data/AGCN2s_ntucs_joint.pdparams \
                                -o inference/AGCN2s_ntucs_joint
```

To get model architecture file `AGCN2s_ntucs_joint.pdmodel` and parameters file `AGCN2s_ntucs_joint.pdiparams`.

- Args usage please refer to [Model Inference](https://github.com/PaddlePaddle/PaddleVideo/blob/release/2.0/docs/zh-CN/start.md#2-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86).

### infer

```bash
python3.7 tools/predict.py --input_file data/example_NTU-RGB-D_sketeton.npy \
                           --config configs/recognition/agcn2s/2sagcn_ntucs_joint.yaml \
                           --model_file inference/AGCN2s_ntucs_joint/AGCN2s_ntucs_joint.pdmodel \
                           --params_file inference/AGCN2s_ntucs_joint/AGCN2s_ntucs_joint.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
```

## Reference

- [Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf), Lei Shi and Yifan Zhang and Jian Cheng and Hanqing Lu

