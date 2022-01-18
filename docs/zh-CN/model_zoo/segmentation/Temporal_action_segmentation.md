[English](../../../en/model_zoo/segmentation/Temporal_action_segmentation.md) | 简体中文

本仓库提供经典和热门时序动作分割模型的性能和精度对比

| Model | Metrics | Value | Flops(M) |Params(M) | test time(ms) bs=1 | test time(ms) bs=2 | inference time(ms) bs=1 | inference time(ms) bs=2 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MS-TCN | F1@0.5 | 38.8% | 791.360 | 0.8 | 170 | - | 10.68 | - |
| ASRF | F1@0.5 | 55.7% | 1,283.328 | 1.3 | 190 | - | 16.34 | - |

* 模型名称：填写模型的具体名字，比如PP-TSM
* Metrics：填写模型测试时所用的指标，使用的数据集为**breakfast**
* Value：填写Metrics指标对应的数值，一般保留小数点后两位
* Flops(M)：模型一次前向运算所需的浮点运算量，可以调用PaddleVideo/tools/summary.py脚本计算（不同模型可能需要稍作修改），保留小数点后一位，使用数据**输入形状为(1, 2048, 1000)的张量**测得
* Params(M)：模型参数量，和Flops一起会被脚本计算出来，保留小数点后一位
* test time(ms) bs=1：python脚本开batchsize=1测试时，一个样本所需的耗时，保留小数点后两位。测试使用的数据集为**breakfast**。
* test time(ms) bs=2：python脚本开batchsize=2测试时，一个样本所需的耗时，保留小数点后两位。时序动作分割模型一般是全卷积网络，所以训练、测试和推理的batch_size都是1。测试使用的数据集为**breakfast**。
* inference time(ms) bs=1：推理模型用GPU（默认V100）开batchsize=1测试时，一个样本所需的耗时，保留小数点后两位。推理使用的数据集为**breakfast**。
* inference time(ms) bs=2：推理模型用GPU（默认V100）开batchsize=1测试时，一个样本所需的耗时，保留小数点后两位。时序动作分割模型一般是全卷积网络，所以训练、测试和推理的batch_size都是1。推理使用的数据集为**breakfast**。
