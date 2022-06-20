
# 飞桨训推一体认证（TIPC）

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了PaddleOCR中所有模型的飞桨训推一体认证 (Training and Inference Pipeline Certification(TIPC)) 信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。

![[tipc导图](https://github.com/ELKYang/2s-AGCN-paddle/blob/main/test_tipc/imgs/guide.png)](https://github.com/PaddlePaddle/models/raw/release/2.2/tutorials/tipc/images/tipc_guide.png)

## 2. 测试工具简介
### 目录介绍

```shell
├──test_tipc/
  ├── configs/                        # 配置文件目录
      ├── EfficientGCNB0              # EfficientGCNB0 模型的测试配置文件目录 
          ├── train_infer_python.txt  # 测试Linux上python训练预测（基础训练预测）的配置文件
  ├── common_func.sh
  ├── prepare.sh                      # 完成test_*.sh运行所需要的数据和模型下载
  ├── test_train_inference_python.sh  # 测试python训练预测的主程序
  └── READEME.md                      # 使用文档
├──log                                #日志和模型保存路径
```

### 测试流程概述

使用本工具，可以测试不同功能的支持情况，以及预测结果是否对齐，测试流程概括如下：

1. 运行prepare.sh准备测试所需数据和模型；
2. 运行要测试的功能对应的测试脚本`test_train_inference_python.sh`，产出log，由log可以看到不同配置是否运行成功；

测试单项功能仅需两行命令，**如需测试不同模型/功能，替换配置文件即可**，命令格式如下：
```shell
# 功能：准备数据
# 格式：bash + 运行脚本 + 参数1: 源数据目录 + 参数2: 保存路径 + 参数3： 生成训练数据/验证集
bash test_tipc/prepare.sh [source_file_directory] [save_directory] [train/eval]
# 功能：运行测试
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/test_train_inference_python.sh configs/[model_name]/[params_file_name]  [Mode]
```

以下为示例：
```shell
# 功能：准备数据
# 格式：bash + 运行脚本 + 参数1: 源数据目录 + 参数2: 保存路径 + 参数3： 生成训练数据/验证集
bash test_tipc/prepare.sh ./data/npy_dataset ./data/ntu/tiny_dataset train
# 功能：运行测试
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/EfficientGCNB0/train_infer_python.txt lite_train_lite_infer
```

### TIPC过程log
##### 训练


    [ 2022-06-06 17:45:16,583 ] Saving folder path: ./log/EfficinetGCNB0/lite_train_lite_infer/norm_train_gpus_0/2001_infer_EfficientGCN-B0_ntu-xsub/2022-06-06 17-45-16
    [ 2022-06-06 17:45:16,583 ] 
    [ 2022-06-06 17:45:16,583 ] Starting preparing ...
    [ 2022-06-06 17:45:16,587 ] Saving folder path: ./log/EfficinetGCNB0/lite_train_lite_infer/norm_train_gpus_0/2001_infer_EfficientGCN-B0_ntu-xsub/2022-06-06 17-45-16
    [ 2022-06-06 17:45:16,587 ] Saving model name: None
    [ 2022-06-06 17:45:16,593 ] Dataset: ntu-xsub
    [ 2022-06-06 17:45:16,593 ] Batch size: train-16, eval-16
    [ 2022-06-06 17:45:16,593 ] Data shape (branch, channel, frame, joint, person): [3, 6, 288, 25, 2]
    [ 2022-06-06 17:45:16,593 ] Number of action classes: 60
    W0606 17:45:17.656214 22933 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
    W0606 17:45:17.659608 22933 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    [ 2022-06-06 17:45:19,556 ] Model: EfficientGCN-B0 {'stem_channel': 64, 'block_args': [[48, 1, 0.5], [24, 1, 0.5], [64, 2, 1], [128, 2, 1]], 'fusion_stage': 2, 'act_type': 'swish', 'att_type': 'stja', 'layer_type': 'SG', 'drop_prob': 0.25, 'kernel_size': [5, 2], 'scale_args': [1.2, 1.35], 'expand_ratio': 0, 'reduct_ratio': 2, 'bias': True, 'edge': True}
    [ 2022-06-06 17:45:19,558 ] LR_Scheduler: cosine {'max_epoch': 1, 'warm_up': 10}
    [ 2022-06-06 17:45:19,559 ] Optimizer: Momentum {'learning_rate': <paddle.optimizer.lr.LambdaDecay object at 0x7fa4bdb30510>, 'momentum': 0.9, 'use_nesterov': True, 'weight_decay': 0.0001}
    [ 2022-06-06 17:45:19,560 ] Loss function: CrossEntropyLoss
    [ 2022-06-06 17:45:19,560 ] Successful!
    [ 2022-06-06 17:45:19,560 ] 
    [ 2022-06-06 17:45:19,560 ] Starting training ...
    Loss: 4.6621, LR: 0.0100: 100%|█████████████████████████████████████████████████████████████████████████| 50/50 [00:17<00:00,  2.80it/s]
    [ 2022-06-06 17:45:37,437 ] Epoch: 1/1, Training accuracy: 16/800(2.00%), Training time: 17.88s
    [ 2022-06-06 17:45:37,437 ] 
    [ 2022-06-06 17:45:37,437 ] Evaluating for epoch 1/1 ...
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  7.69it/s]
    [ 2022-06-06 17:45:43,942 ] Top-1 accuracy: 19/800(2.38%), Top-5 accuracy: 79/800(9.88%), Mean loss:4.6169
    [ 2022-06-06 17:45:43,942 ] Evaluating time: 6.50s, Speed: 123.02 sequnces/(second*GPU)
    [ 2022-06-06 17:45:43,943 ] 
    [ 2022-06-06 17:45:43,943 ] Saving model for epoch 1/1 ...
    [ 2022-06-06 17:45:43,973 ] Best top-1 accuracy: 2.38%, Total time: 00d-00h-00m-24s
    [ 2022-06-06 17:45:43,973 ] 
    [ 2022-06-06 17:45:43,973 ] Finish training!
    [ 2022-06-06 17:45:43,973 ] 
     Run successfully with command - python main.py -c 2001_infer --work_dir=./log/EfficinetGCNB0/lite_train_lite_infer/norm_train_gpus_0      !

  

##### 验证


    [ 2022-06-06 17:45:46,807 ] Saving folder path: ./workdir/temp
    [ 2022-06-06 17:45:46,807 ] 
    [ 2022-06-06 17:45:46,807 ] Starting preparing ...
    [ 2022-06-06 17:45:46,811 ] Saving folder path: ./workdir/temp
    [ 2022-06-06 17:45:46,811 ] Saving model name: ./log/EfficinetGCNB0/lite_train_lite_infer/norm_train_gpus_0/model.pdparams
    [ 2022-06-06 17:45:46,817 ] Dataset: ntu-xsub
    [ 2022-06-06 17:45:46,817 ] Batch size: train-16, eval-16
    [ 2022-06-06 17:45:46,817 ] Data shape (branch, channel, frame, joint, person): [3, 6, 288, 25, 2]
    [ 2022-06-06 17:45:46,817 ] Number of action classes: 60
    W0606 17:45:47.879998 22982 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
    W0606 17:45:47.883787 22982 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    [ 2022-06-06 17:45:49,813 ] Model: EfficientGCN-B0 {'stem_channel': 64, 'block_args': [[48, 1, 0.5], [24, 1, 0.5], [64, 2, 1], [128, 2, 1]], 'fusion_stage': 2, 'act_type': 'swish', 'att_type': 'stja', 'layer_type': 'SG', 'drop_prob': 0.25, 'kernel_size': [5, 2], 'scale_args': [1.2, 1.35], 'expand_ratio': 0, 'reduct_ratio': 2, 'bias': True, 'edge': True}
    [ 2022-06-06 17:45:49,815 ] LR_Scheduler: cosine {'max_epoch': 1, 'warm_up': 10}
    [ 2022-06-06 17:45:49,816 ] Optimizer: Momentum {'learning_rate': <paddle.optimizer.lr.LambdaDecay object at 0x7fab8136fc50>, 'momentum': 0.9, 'use_nesterov': True, 'weight_decay': 0.0001}
    [ 2022-06-06 17:45:49,816 ] Loss function: CrossEntropyLoss
    [ 2022-06-06 17:45:49,817 ] Successful!
    [ 2022-06-06 17:45:49,817 ] 
    [ 2022-06-06 17:45:49,817 ] Loading evaluating model ...
    [ 2022-06-06 17:45:49,869 ] Successful!
    [ 2022-06-06 17:45:49,869 ] 
    [ 2022-06-06 17:45:49,869 ] Starting evaluating ...
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  7.48it/s]
    [ 2022-06-06 17:45:56,561 ] Top-1 accuracy: 19/800(2.38%), Top-5 accuracy: 79/800(9.88%), Mean loss:4.6169
    [ 2022-06-06 17:45:56,561 ] Evaluating time: 6.69s, Speed: 119.60 sequnces/(second*GPU)
    [ 2022-06-06 17:45:56,562 ] 
    [ 2022-06-06 17:45:56,562 ] Finish evaluating!
     Run successfully with command - python main.py -e -c 2001_infer --pretrained=./log/EfficinetGCNB0/lite_train_lite_infer/norm_train_gpus_0/model.pdparams!   


##### 模型导出
  
    [ 2022-06-06 17:45:59,311 ] Saving folder path: ./workdir/temp
    [ 2022-06-06 17:45:59,311 ] 
    [ 2022-06-06 17:45:59,311 ] Starting preparing ...
    [ 2022-06-06 17:45:59,315 ] Saving folder path: ./workdir/temp
    [ 2022-06-06 17:45:59,316 ] Saving model name: ./log/EfficinetGCNB0/lite_train_lite_infer/norm_train_gpus_0/model.pdparams
    [ 2022-06-06 17:45:59,321 ] Dataset: ntu-xsub
    [ 2022-06-06 17:45:59,321 ] Batch size: train-16, eval-16
    [ 2022-06-06 17:45:59,321 ] Data shape (branch, channel, frame, joint, person): [3, 6, 288, 25, 2]
    [ 2022-06-06 17:45:59,321 ] Number of action classes: 60
    W0606 17:46:00.405385 23074 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
    W0606 17:46:00.409031 23074 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    [ 2022-06-06 17:46:02,315 ] Model: EfficientGCN-B0 {'stem_channel': 64, 'block_args': [[48, 1, 0.5], [24, 1, 0.5], [64, 2, 1], [128, 2, 1]], 'fusion_stage': 2, 'act_type': 'swish', 'att_type': 'stja', 'layer_type': 'SG', 'drop_prob': 0.25, 'kernel_size': [5, 2], 'scale_args': [1.2, 1.35], 'expand_ratio': 0, 'reduct_ratio': 2, 'bias': True, 'edge': True}
    [ 2022-06-06 17:46:02,317 ] LR_Scheduler: cosine {'max_epoch': 1, 'warm_up': 10}
    [ 2022-06-06 17:46:02,318 ] Optimizer: Momentum {'learning_rate': <paddle.optimizer.lr.LambdaDecay object at 0x7f67159b3d50>, 'momentum': 0.9, 'use_nesterov': True, 'weight_decay': 0.0001}
    [ 2022-06-06 17:46:02,319 ] Loss function: CrossEntropyLoss
    [ 2022-06-06 17:46:02,319 ] Successful!
    [ 2022-06-06 17:46:02,319 ] 
    [ 2022-06-06 17:46:02,319 ] Loading model ...
    [ 2022-06-06 17:46:06,090 ] Successful!
    [ 2022-06-06 17:46:06,090 ] 
     Run successfully with command - python main.py -c 2001_infer -ex  --pretrained=./log/EfficinetGCNB0/lite_train_lite_infer/norm_train_gpus_0/model.pdparams!  
  
##### GPU/CPU 推理
[log](https://github.com/Wuxiao85/paddle_EfficientGCNv/tree/main/log/EfficinetGCNB0/lite_train_lite_infer)
  
  
  
