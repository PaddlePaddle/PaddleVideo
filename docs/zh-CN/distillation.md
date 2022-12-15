# 知识蒸馏

---
## 目录

- [1. 知识蒸馏简介](#1)
    - [1.1 Response based distillation](#1.1)
    - [1.2 Feature based distillation](#1.2)
    - [1.3 Relation based distillation](#1.3)
- [2. PaddleVideo支持的知识蒸馏算法](#2)
    - [2.1 DML](#2.1)
- [3. 参考文献](#3)


<a name="1"></a>

### 1. 知识蒸馏简介

近年来，深度神经网络在计算机视觉、自然语言处理等领域被验证是一种极其有效的解决问题的方法。通过构建合适的神经网络，加以训练，最终网络模型的性能指标基本上都会超过传统算法。

在数据量足够大的情况下，通过合理构建网络模型的方式增加其参数量，可以显著改善模型性能，但是这又带来了模型复杂度急剧提升的问题。大模型在实际场景中使用的成本较高。

深度神经网络一般有较多的参数冗余，目前有几种主要的方法对模型进行压缩，减小其参数量。如裁剪、量化、知识蒸馏等，其中知识蒸馏是指使用教师模型(teacher model)去指导学生模型(student model)学习特定任务，保证小模型在参数量不变的情况下，得到比较大的性能提升，甚至获得与大模型相似的精度指标 [1]。

根据蒸馏方式的不同，可以将知识蒸馏方法分为3个不同的类别：Response based distillation、Feature based distillation、Relation based distillation。下面进行详细介绍。

<a name='1.1'></a>

#### 1.1 Response based distillation

最早的知识蒸馏算法 KD，由 Hinton 提出，训练的损失函数中除了 gt loss 之外，还引入了学生模型与教师模型输出的 KL 散度，最终精度超过单纯使用 gt loss 训练的精度。这里需要注意的是，在训练的时候，需要首先训练得到一个更大的教师模型，来指导学生模型的训练过程。

上述标准的蒸馏方法是通过一个大模型作为教师模型来指导学生模型提升效果，而后来又发展出 DML(Deep Mutual Learning)互学习蒸馏方法 [7]，即通过两个结构相同的模型互相学习。具体的。相比于 KD 等依赖于大的教师模型的知识蒸馏算法，DML 脱离了对大的教师模型的依赖，蒸馏训练的流程更加简单，模型产出效率也要更高一些。

<a name='1.2'></a>

#### 1.2 Feature based distillation

Heo 等人提出了 OverHaul [8], 计算学生模型与教师模型的 feature map distance，作为蒸馏的 loss，在这里使用了学生模型、教师模型的转移，来保证二者的 feature map 可以正常地进行 distance 的计算。

基于 feature map distance 的知识蒸馏方法也能够和 `1.1 章节` 中的基于 response 的知识蒸馏算法融合在一起，同时对学生模型的输出结果和中间层 feature map 进行监督。而对于 DML 方法来说，这种融合过程更为简单，因为不需要对学生和教师模型的 feature map 进行转换，便可以完成对齐(alignment)过程。P

<a name='1.3'></a>

#### 1.3 Relation based distillation

[1.1](#1.1) 和 [1.2](#1.2) 章节中的论文中主要是考虑到学生模型与教师模型的输出或者中间层 feature map，这些知识蒸馏算法只关注个体的输出结果，没有考虑到个体之间的输出关系。

Park 等人提出了 RKD [10]，基于关系的知识蒸馏算法，RKD 中进一步考虑个体输出之间的关系，使用 2 种损失函数，二阶的距离损失（distance-wise）和三阶的角度损失（angle-wise）

本论文提出的算法关系知识蒸馏（RKD）迁移教师模型得到的输出结果间的结构化关系给学生模型，不同于之前的只关注个体输出结果，RKD 算法使用两种损失函数：二阶的距离损失(distance-wise)和三阶的角度损失(angle-wise)。在最终计算蒸馏损失函数的时候，同时考虑 KD loss 和 RKD loss。最终精度优于单独使用 KD loss 蒸馏得到的模型精度。

<a name='2'></a>

### 2. PaddleVideo支持的知识蒸馏算法

#### 2.1 DML

##### 2.1.1 DML 算法介绍

论文信息：

> [Deep Mutual Learning](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.html)
>
> Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu
>
> CVPR, 2018

DML论文中，在蒸馏的过程中，不依赖于教师模型，两个结构相同的模型互相学习，计算彼此输出（logits）的KL散度，最终完成训练过程。


在Kinetics-400公开数据集上，效果如下所示。

| 策略 | 骨干网络 | 配置文件 | Top-1 acc |
| --- | --- | --- | --- |
| baseline | PP-TSMv2 | [pptsm_lcnet_k400_frames_uniform.yaml](../configs/recognition/pptsm/v2/pptsm_lcnet_k400_frames_uniform.yaml) | 73.1% |
| DML | PP-TSMv2 | [pptsm_lcnet_k400_frames_uniform_dml_distillation.yaml](../configs/recognition/pptsm/v2/pptsm_lcnet_k400_frames_uniform_dml_distillation.yaml) | 74.38%(**+1.28%**) |


##### 2.1.2 DML 配置

DML配置如下所示。在模型构建Arch字段中，需要同时定义学生模型与教师模型，教师模型与学生模型均保持梯度更新状态。在损失函数Loss字段中，需要定义`DistillationDMLLoss`（学生与教师之间的JS-Div loss）以及`DistillationCELoss`（学生与教师关于真值标签的CE loss），作为训练的损失函数。


使用蒸馏训练，配置文件需要做一定的修改：
原始Student模型训练配置文件：
```yaml
MODEL:
    framework: "Recognizer2D"
    backbone:
        name: "PPTSM_v2"
        pretrained: "data/PPLCNetV2_base_ssld_pretrained.pdparams"
        num_seg: 16
    head:
        name: "MoViNetHead"
```

DML配置如下所示。在模型构建MODEL字段中，需要指定framework为`RecognizerDistillation`，同时定义学生模型与教师模型，教师模型与学生模型均保持梯度更新状态。在损失函数Loss字段中，需要定义`DistillationDMLLoss`（学生与教师之间的JS-Div loss）以及`DistillationCELoss`（学生与教师关于真值标签的CE loss），作为训练的损失函数。

```yaml
MODEL:
    framework: "RecognizerDistillation"
    freeze_params_list:
    - False # Teacher是否可学习
    - False # Student是否可学习
    models:
    - Teacher: # 指定Teacher模型
        backbone:
            name: "ResNetTweaksTSM" #Teacher模型名称
            pretrained: "data/ResNet50_vd_ssld_v2_pretrained.pdparams"
            depth: 50
            num_seg: 16
        head:
            name: "ppTSMHead" # Teacher模型head
            num_classes: 400
            in_channels: 2048
            drop_ratio: 0.5
            std: 0.01
            num_seg: 16
    - Student:
        backbone: # 指定Student模型
            name: "PPTSM_v2" #Student模型名称
            pretrained: "data/PPLCNetV2_base_ssld_pretrained.pdparams"
            num_seg: 16
        head:
            name: "MoViNetHead" # Student模型head
    loss: # 指定蒸馏loss
        Train:  # 训练时loss计算
            - name: "DistillationCELoss" # 蒸馏损失1
              model_name_pairs: ["Student", "GroundTruth"] # 计算loss的对象
            - name: "DistillationCELoss" # 蒸馏损失2
              model_name_pairs: ["Teacher", "GroundTruth"]
            - name: "DistillationDMLLoss" # 蒸馏损失3
              model_name_pairs: ["Student", "Teacher"]
        Val:   # 评估时loss计算
            - name: "DistillationCELoss"
              model_name_pairs: ["Student", "GroundTruth"]
```

若将教师模型设置为Student自身，便是一种简单的自蒸馏方式，示例配置文件如下：
```yaml
MODEL:
    framework: "RecognizerDistillation"
    freeze_params_list:
    - False # Teacher是否可学习
    - False # Student是否可学习
    models:
    - Teacher: # 指定Teacher模型
        backbone:
            name: "PPTSM_v2"
            pretrained: "data/PPLCNetV2_base_ssld_pretrained.pdparams"
            num_seg: 16
        head:
            name: "MoViNetHead"
    - Student:
        backbone: # 指定Student模型
            name: "PPTSM_v2"
            pretrained: "data/PPLCNetV2_base_ssld_pretrained.pdparams"
            num_seg: 16
        head:
            name: "MoViNetHead"
    loss: # 指定蒸馏loss
        Train:  # 训练时loss计算
            - name: "DistillationCELoss" # 蒸馏损失1
              model_name_pairs: ["Student", "GroundTruth"] # 计算loss的对象
            - name: "DistillationCELoss" # 蒸馏损失2
              model_name_pairs: ["Teacher", "GroundTruth"]
            - name: "DistillationDMLLoss" # 蒸馏损失3
              model_name_pairs: ["Student", "Teacher"]
        Val:   # 评估时loss计算
            - name: "DistillationCELoss"
              model_name_pairs: ["Student", "GroundTruth"]
```

实验发现，在Kinetics-400公开数据集上，使用自蒸馏方法，PP-TSMv2的精度也能获得1个点左右的提升:

| 策略 | 教师网络 | Top-1 acc |
| --- | --- | --- |
| baseline | - | 69.06% |
| DML | PP-TSMv2 | 70.34%(**+1.28%**) |
| DML | PP-TSM_ResNet50 | 71.27%(**+2.20%**) |

* 注：完整的PP-TSMv2加了其它trick训练，这里为了方便对比，baseline未加其它tricks，因此指标比官网最终开源出来的模型精度低一些。

完成配置文件的修改后，参考[使用说明](./usage.md)即可开启模型训练、测试与推理。


<a name="3"></a>

## 3. 参考文献

[1] Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network[J]. arXiv preprint arXiv:1503.02531, 2015.

[2] Bagherinezhad H, Horton M, Rastegari M, et al. Label refinery: Improving imagenet classification through label progression[J]. arXiv preprint arXiv:1805.02641, 2018.

[3] Yalniz I Z, Jégou H, Chen K, et al. Billion-scale semi-supervised learning for image classification[J]. arXiv preprint arXiv:1905.00546, 2019.

[4] Cubuk E D, Zoph B, Mane D, et al. Autoaugment: Learning augmentation strategies from data[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2019: 113-123.

[5] Touvron H, Vedaldi A, Douze M, et al. Fixing the train-test resolution discrepancy[C]//Advances in Neural Information Processing Systems. 2019: 8250-8260.

[6] Cui C, Guo R, Du Y, et al. Beyond Self-Supervision: A Simple Yet Effective Network Distillation Alternative to Improve Backbones[J]. arXiv preprint arXiv:2103.05959, 2021.

[7] Zhang Y, Xiang T, Hospedales T M, et al. Deep mutual learning[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4320-4328.

[8] Heo B, Kim J, Yun S, et al. A comprehensive overhaul of feature distillation[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 1921-1930.

[9] Du Y, Li C, Guo R, et al. PP-OCRv2: Bag of Tricks for Ultra Lightweight OCR System[J]. arXiv preprint arXiv:2109.03144, 2021.

[10] Park W, Kim D, Lu Y, et al. Relational knowledge distillation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 3967-3976.

[11] Zhao B, Cui Q, Song R, et al. Decoupled Knowledge Distillation[J]. arXiv preprint arXiv:2203.08679, 2022.

[12] Ji M, Heo B, Park S. Show, attend and distill: Knowledge distillation via attention-based feature matching[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2021, 35(9): 7945-7952.

[13] Huang T, You S, Wang F, et al. Knowledge Distillation from A Stronger Teacher[J]. arXiv preprint arXiv:2205.10536, 2022.

[14] https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/advanced_tutorials/knowledge_distillation.md#1.1.2
