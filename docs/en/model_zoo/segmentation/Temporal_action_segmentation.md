[简体中文](../../../zh-CN/model_zoo/segmentation/Temporal_action_segmentation.md) | English

This repo provides performance and accuracy comparison between classical and popular sequential action segmentation models

| Model | Metrics | Value | Flops(M) |Params(M) | test time(ms) bs=1 | test time(ms) bs=2 | inference time(ms) bs=1 | inference time(ms) bs=2 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MS-TCN | F1@0.5 | 38.8% | 791.360 | 0.8 | 170 | - | 10.68 | - |
| ASRF | F1@0.5 | 55.7% | 1,283.328 | 1.3 | 190 | - | 16.34 | - |

* Model: model name, for example: PP-TSM
* Metrics: Fill in the indicators used in the model test, and the data set used is **breakfast**
* Value: Fill in the value corresponding to the metrics index, and generally keep two decimal places
* Flops(M): The floating-point computation required for one forward operation of the model can be called `paddlevideo/tools/summary.py`script calculation (different models may need to be modified slightly), keep one decimal place, and measure it with data **input tensor with shape of (1, 2048, 1000)**
* Params(M): The model parameter quantity, together with flops, will be calculated by the script, and one decimal place will be reserved
* test time(ms) bs=1: When the python script starts the batchsize = 1 test, the time required for a sample is kept to two decimal places. The data set used in the test is **breakfast**.
* test time(ms) bs=2: When the python script starts the batchsize = 2 test, the time required for a sample is kept to two decimal places. The sequential action segmentation model is generally a full convolution network, so the batch of training, testing and reasoning_ Size is 1. The data set used in the test is **breakfast**.
* inference time(ms) bs=1: When the reasoning model is tested with GPU (default V100) with batchsize = 1, the time required for a sample is reserved to two decimal places. The dataset used for reasoning is **breakfast**.
* inference time(ms) bs=2: When the reasoning model is tested with GPU (default V100) with batchsize = 1, the time required for a sample is reserved to two decimal places. The sequential action segmentation model is generally a full convolution network, so the batch of training, testing and reasoning_ Size is 1. The dataset used for reasoning is **breakfast**.
