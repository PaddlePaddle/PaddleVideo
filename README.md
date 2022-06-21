# EfficientGCN_paddle
## 1.简介
This is an unofficial code based on PaddlePaddle of IEEE 2022 paper:

[EfficientGCN: Constructing Stronger and Faster Baselines for Skeleton-based Action Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9729609)




![image](https://user-images.githubusercontent.com/59130750/171786592-c1d7d67c-7d0c-4816-b8ff-7f7a5fe0320b.png)

这是一篇骨骼点动作识别领域的文章，文章提出了EfficientGCN模型，该模型在MIB网络中结合可分离的卷积层，利用图卷积网络对视频动作进行识别，骨骼点数据相对于传统RGB数据更具解释性与鲁棒性。该方法相较于传统中参数量较大的双流特征提取方式，在模型的前端选择融合三个输入分支并输入主流模型提取特征，通过这种方式减小了模型的复杂度。  
论文地址：[EfficientGCN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9729609)  
原论文代码地址：[EfficientGCN Code](https://gitee.com/yfsong0709/EfficientGCNv1)


 
## 2.复现精度

注：NTU RGB+D 60数据集,EfficientGCN-B0模型下的x-sub和x-view分别对应2001和2002模型

 NTU RGB+D 60数据集，EfficientGCN-B0模型  | X-sub（2001）  | X-view （2002） 
 ---- | ----- | ------  
 Paper  | 90.2% | 94.9% 
 Paddle  | 90.2% | 94.99%  
 
 在NTU RGB+D 60数据集上基本达到验收标准  
 训练日志和模型权重：  https://github.com/small-whirlwind/EfficientGCN_paddle/tree/main/workdir_pad

 aistudio的实现方式：  
 在./tasks/文件夹内运行CUDA_VISIBLE_DEVICES=0 python3 main.py --gpus 0 -c 2001 -e  
 在./tasks/文件夹内运行CUDA_VISIBLE_DEVICES=0 python3 main.py --gpus 0 -c 2002 -e  
 选择要测试的模型即可  


## 3.环境依赖
- 硬件：GeForce RTX 2080 Ti  
- Based on Python3 (anaconda, >= 3.5) and PyTorch (>= 1.6.0).
- paddlePaddle-gpu==2.2.2  

## 4.数据集和预训练模型下载
复现任务是在NTU RGB+D 60数据集上进行的，只需要骨骼点1-17的部分，可以从这里下载https://drive.google.com/file/d/1CUZnBtYwifVXS21yVg62T-vrPVayso5H/view  

预训练模型，在这里下载https://drive.google.com/drive/folders/1HpvkKyfmmOCzuJXemtDxQCgGGQmWMvj4 。在本次任务中，下载2001,2002即可。

但此处但此处下载的ckpy文件适配于pytorch框架，在此给出两种解决方案：  
- 直接使用项目pretrained文件夹中转换好的ckpy  
- 通过本项目中的transferForPth.py文件进行模型转换，将.pth文件转换为适配paddle的.pdparams文件。

依次运行如下命令以解压：


```
unzip /home/aistudio/data/data150382/original.zip -d /home/aistudio/dataset/data/npy_dataset

unzip /home/aistudio/data/data150382/transformed.zip -d /home/aistudio/dataset/data/npy_dataset

tar -xvf /home/aistudio/data/data150382/xsub.tar -C /home/aistudio/dataset/

tar -xvf /home/aistudio/data/data150382/xview.tar -C /home/aistudio/dataset/

tar -xvf /home/aistudio/data/data150382/nturgbd_skeletons_s001_to_s017.tar -C /home/aistudio/dataset/
```

## 5.数据预处理
### 5.1 config文件生成
输入数据集路径、预处理后的数据集存放路径、预训练模型路径等，生成config文件 
```
python scripts/modify_configs.py --root_folder <path/to/save/numpy/data> --ntu60_path <path/to/ntu60/dataset> --ntu120_path <path/to/ntu120/dataset> --pretrained_path <path/to/save/pretraiined/model> --work_dir <path/to/work/dir>
```
示例：
进入/home/aistudio/EfficientGCN_paddle文件夹，运行
```
python .\PaddleVideo-develop\paddlevideo\utils/modify_configs.py --root_folder .\PaddleVideo-develop\data\ntu\npy_dataset\ --ntu60-path .\PaddleVideo-develop\data\ntu\nturgbd_skeletons_s001_to_s017\ --ntu120_path /share/NTU-RGB-D120 --pretrained_path .\PaddleVideo-develop\pretrained  --work_dir .\PaddleVideo-develop\workdir_pad --gpus 0
```
### 5.2 数据预处理
进入/home/aistudio/EfficientGCN_paddle/tasks/文件夹
(由于我们提供的数据集中已经包含了预处理之后的文件，故而可以不执行此步，直接解压original.zip，transformed.zip, xsub.zip，xview.zip，而无需下载、解压作为raw data的nturgbd_skeletons_s001_to_s017.tar)

```
python ./PaddleVideo-develop/paddlevideo/tasks/main.py -c 2001 -gd -np

python ./PaddleVideo-develop/paddlevideo/tasks/main.py -c 2002 -gd -np
```
最终，经过预处理后的数据集文件夹格式如下所示：
```
.\PaddleVideo-develop\data\ntu
                               |-xview
                               |-xsub
                               |-npy_datatset |-transformed
                                              |-original
                               |-nturgbd_skeletons_s001_to_s017


```
## 5 模型训练
在终端输入如下命令行进行训练：
```
python main.py -c <config>
```
在本次复现项目中，针对2001,2002两个model，输入

x-sub(2001)
```
export CUDA_VISIBLE_DEVICES=0
python ./PaddleVideo-develop/paddlevideo/tasks/main.py --gpus 0 -c 2001
```
对于x-view(2002)

```
export CUDA_VISIBLE_DEVICES=0
python ./PaddleVideo-develop/paddlevideo/tasks/main.py --gpus 0 -c 2002
```


部分训练输出如下（2001）：

```
[ 2022-06-14 08:48:56,173 ] Evaluating for epoch 78/90 ...
[ 2022-06-14 08:51:29,527 ] Top-1 accuracy: 14606/16480(88.63%), Top-5 accuracy: 16193/16480(98.26%), Mean loss:0.4205
[ 2022-06-14 08:51:29,527 ] Evaluating time: 153.35s, Speed: 107.46 sequnces/(second*GPU)
[ 2022-06-14 08:51:29,527 ] 
[ 2022-06-14 08:51:29,569 ] Saving model for epoch 78/90 ...
[ 2022-06-14 08:51:29,589 ] Best top-1 accuracy: 90.20%, Total time: 00d-22h-56m-04s
[ 2022-06-14 08:51:29,589 ] 
[ 2022-06-14 09:08:07,164 ] Epoch: 79/90, Training accuracy: 39772/40080(99.23%), Training time: 997.57s
[ 2022-06-14 09:08:07,164 ] 
[ 2022-06-14 09:08:07,165 ] Evaluating for epoch 79/90 ...
[ 2022-06-14 09:10:40,749 ] Top-1 accuracy: 14582/16480(88.48%), Top-5 accuracy: 16159/16480(98.05%), Mean loss:0.4521
[ 2022-06-14 09:10:40,749 ] Evaluating time: 153.58s, Speed: 107.30 sequnces/(second*GPU)
[ 2022-06-14 09:10:40,749 ] 
[ 2022-06-14 09:10:40,808 ] Saving model for epoch 79/90 ...
[ 2022-06-14 09:10:40,828 ] Best top-1 accuracy: 90.20%, Total time: 00d-23h-15m-15s
[ 2022-06-14 09:10:40,829 ] 
[ 2022-06-14 09:27:18,213 ] Epoch: 80/90, Training accuracy: 39553/40080(98.69%), Training time: 997.38s
[ 2022-06-14 09:27:18,214 ] 
[ 2022-06-14 09:27:18,214 ] Evaluating for epoch 80/90 ...
[ 2022-06-14 09:29:51,722 ] Top-1 accuracy: 14441/16480(87.63%), Top-5 accuracy: 16149/16480(97.99%), Mean loss:0.4696
[ 2022-06-14 09:29:51,723 ] Evaluating time: 153.51s, Speed: 107.36 sequnces/(second*GPU)
[ 2022-06-14 09:29:51,723 ] 
[ 2022-06-14 09:29:51,771 ] Saving model for epoch 80/90 ...
[ 2022-06-14 09:29:51,790 ] Best top-1 accuracy: 90.20%, Total time: 00d-23h-34m-26s
```
部分训练输出如下（2002）：
```
[ 2022-06-16 07:56:26,019 ] Saving model for epoch 70/80 ...
[ 2022-06-16 07:56:26,040 ] Best top-1 accuracy: 94.94%, Total time: 00d-19h-21m-06s
[ 2022-06-16 07:56:26,040 ] 
[ 2022-06-16 08:12:00,023 ] Epoch: 71/80, Training accuracy: 37372/37632(99.31%), Training time: 933.98s
[ 2022-06-16 08:12:00,023 ] 
[ 2022-06-16 08:12:00,023 ] Evaluating for epoch 71/80 ...
[ 2022-06-16 08:14:55,711 ] Top-1 accuracy: 17918/18928(94.66%), Top-5 accuracy: 18781/18928(99.22%), Mean loss:0.2032
[ 2022-06-16 08:14:55,711 ] Evaluating time: 175.69s, Speed: 107.74 sequnces/(second*GPU)
[ 2022-06-16 08:14:55,711 ] 
[ 2022-06-16 08:14:55,763 ] Saving model for epoch 71/80 ...
[ 2022-06-16 08:14:55,785 ] Best top-1 accuracy: 94.94%, Total time: 00d-19h-39m-36s
[ 2022-06-16 08:14:55,785 ] 
[ 2022-06-16 08:30:30,106 ] Epoch: 72/80, Training accuracy: 37378/37632(99.33%), Training time: 934.32s
[ 2022-06-16 08:30:30,106 ] 
[ 2022-06-16 08:30:30,107 ] Evaluating for epoch 72/80 ...
[ 2022-06-16 08:33:26,052 ] Top-1 accuracy: 17907/18928(94.61%), Top-5 accuracy: 18785/18928(99.24%), Mean loss:0.2042
[ 2022-06-16 08:33:26,052 ] Evaluating time: 175.94s, Speed: 107.58 sequnces/(second*GPU)
[ 2022-06-16 08:33:26,052 ] 
[ 2022-06-16 08:33:26,109 ] Saving model for epoch 72/80 ...
[ 2022-06-16 08:33:26,130 ] Best top-1 accuracy: 94.94%, Total time: 00d-19h-58m-06s
[ 2022-06-16 08:33:26,130 ] 
[ 2022-06-16 08:49:00,092 ] Epoch: 73/80, Training accuracy: 37346/37632(99.24%), Training time: 933.96s
[ 2022-06-16 08:49:00,092 ] 
[ 2022-06-16 08:49:00,093 ] Evaluating for epoch 73/80 ...
[ 2022-06-16 08:51:55,775 ] Top-1 accuracy: 17872/18928(94.42%), Top-5 accuracy: 18777/18928(99.20%), Mean loss:0.2032
[ 2022-06-16 08:51:55,775 ] Evaluating time: 175.68s, Speed: 107.74 sequnces/(second*GPU)
[ 2022-06-16 08:51:55,775 ] 
[ 2022-06-16 08:51:55,821 ] Saving model for epoch 73/80 ...
[ 2022-06-16 08:51:55,841 ] Best top-1 accuracy: 94.94%, Total time: 00d-20h-16m-36s
[ 2022-06-16 08:51:55,841 ] 
[ 2022-06-16 09:07:30,096 ] Epoch: 74/80, Training accuracy: 37373/37632(99.31%), Training time: 934.25s
[ 2022-06-16 09:07:30,096 ] 
[ 2022-06-16 09:07:30,097 ] Evaluating for epoch 74/80 ...
[ 2022-06-16 09:10:25,706 ] Top-1 accuracy: 17980/18928(94.99%), Top-5 accuracy: 18794/18928(99.29%), Mean loss:0.1813
[ 2022-06-16 09:10:25,706 ] Evaluating time: 175.61s, Speed: 107.79 sequnces/(second*GPU)
[ 2022-06-16 09:10:25,706 ] 
[ 2022-06-16 09:10:25,746 ] Saving model for epoch 74/80 ...
[ 2022-06-16 09:10:25,770 ] Best top-1 accuracy: 94.99%, Total time: 00d-20h-35m-06s
[ 2022-06-16 09:10:25,771 ] 
[ 2022-06-16 09:26:00,004 ] Epoch: 75/80, Training accuracy: 37316/37632(99.16%), Training time: 934.23s
[ 2022-06-16 09:26:00,004 ] 
[ 2022-06-16 09:26:00,005 ] Evaluating for epoch 75/80 ...
[ 2022-06-16 09:28:55,559 ] Top-1 accuracy: 17878/18928(94.45%), Top-5 accuracy: 18766/18928(99.14%), Mean loss:0.2134
[ 2022-06-16 09:28:55,559 ] Evaluating time: 175.55s, Speed: 107.82 sequnces/(second*GPU)
[ 2022-06-16 09:28:55,559 ] 
[ 2022-06-16 09:28:55,609 ] Saving model for epoch 75/80 ...
[ 2022-06-16 09:28:55,628 ] Best top-1 accuracy: 94.99%, Total time: 00d-20h-53m-36s
[ 2022-06-16 09:28:55,628 ] 
[ 2022-06-16 09:44:29,692 ] Epoch: 76/80, Training accuracy: 37326/37632(99.19%), Training time: 934.06s
[ 2022-06-16 09:44:29,692 ] 
[ 2022-06-16 09:44:29,693 ] Evaluating for epoch 76/80 ...
[ 2022-06-16 09:47:25,533 ] Top-1 accuracy: 17864/18928(94.38%), Top-5 accuracy: 18740/18928(99.01%), Mean loss:0.2088
[ 2022-06-16 09:47:25,533 ] Evaluating time: 175.84s, Speed: 107.64 sequnces/(second*GPU)
[ 2022-06-16 09:47:25,533 ] 
[ 2022-06-16 09:47:25,568 ] Saving model for epoch 76/80 ...
[ 2022-06-16 09:47:25,587 ] Best top-1 accuracy: 94.99%, Total time: 00d-21h-12m-06s
[ 2022-06-16 09:47:25,588 ] 
[ 2022-06-16 10:02:59,796 ] Epoch: 77/80, Training accuracy: 37294/37632(99.10%), Training time: 934.21s
[ 2022-06-16 10:02:59,797 ] 
[ 2022-06-16 10:02:59,797 ] Evaluating for epoch 77/80 ...
[ 2022-06-16 10:05:55,748 ] Top-1 accuracy: 17927/18928(94.71%), Top-5 accuracy: 18787/18928(99.26%), Mean loss:0.1871
[ 2022-06-16 10:05:55,748 ] Evaluating time: 175.95s, Speed: 107.58 sequnces/(second*GPU)
[ 2022-06-16 10:05:55,748 ] 
[ 2022-06-16 10:05:55,793 ] Saving model for epoch 77/80 ...
[ 2022-06-16 10:05:55,825 ] Best top-1 accuracy: 94.99%, Total time: 00d-21h-30m-36s
[ 2022-06-16 10:05:55,825 ] 
[ 2022-06-16 10:21:30,093 ] Epoch: 78/80, Training accuracy: 37261/37632(99.01%), Training time: 934.27s
[ 2022-06-16 10:21:30,093 ] 
[ 2022-06-16 10:21:30,093 ] Evaluating for epoch 78/80 ...
[ 2022-06-16 10:24:26,119 ] Top-1 accuracy: 17834/18928(94.22%), Top-5 accuracy: 18763/18928(99.13%), Mean loss:0.2102
[ 2022-06-16 10:24:26,119 ] Evaluating time: 176.02s, Speed: 107.53 sequnces/(second*GPU)
[ 2022-06-16 10:24:26,119 ] 
[ 2022-06-16 10:24:26,165 ] Saving model for epoch 78/80 ...
[ 2022-06-16 10:24:26,184 ] Best top-1 accuracy: 94.99%, Total time: 00d-21h-49m-06s
[ 2022-06-16 10:24:26,184 ] 
[ 2022-06-16 10:40:00,448 ] Epoch: 79/80, Training accuracy: 37101/37632(98.59%), Training time: 934.26s
[ 2022-06-16 10:40:00,448 ] 
[ 2022-06-16 10:40:00,449 ] Evaluating for epoch 79/80 ...
[ 2022-06-16 10:42:56,801 ] Top-1 accuracy: 17776/18928(93.91%), Top-5 accuracy: 18763/18928(99.13%), Mean loss:0.2192
[ 2022-06-16 10:42:56,801 ] Evaluating time: 176.35s, Speed: 107.33 sequnces/(second*GPU)
[ 2022-06-16 10:42:56,801 ] 
[ 2022-06-16 10:42:56,838 ] Saving model for epoch 79/80 ...
[ 2022-06-16 10:42:56,858 ] Best top-1 accuracy: 94.99%, Total time: 00d-22h-07m-37s
[ 2022-06-16 10:42:56,859 ] 
[ 2022-06-16 10:58:31,100 ] Epoch: 80/80, Training accuracy: 36901/37632(98.06%), Training time: 934.24s
[ 2022-06-16 10:58:31,100 ] 
[ 2022-06-16 10:58:31,101 ] Evaluating for epoch 80/80 ...
[ 2022-06-16 11:01:26,887 ] Top-1 accuracy: 17738/18928(93.71%), Top-5 accuracy: 18752/18928(99.07%), Mean loss:0.2236
[ 2022-06-16 11:01:26,887 ] Evaluating time: 175.79s, Speed: 107.68 sequnces/(second*GPU)
[ 2022-06-16 11:01:26,887 ] 
[ 2022-06-16 11:01:26,924 ] Saving model for epoch 80/80 ...
[ 2022-06-16 11:01:26,945 ] Best top-1 accuracy: 94.99%, Total time: 00d-22h-26m-07s
[ 2022-06-16 11:01:26,946 ] 
[ 2022-06-16 11:01:26,946 ] Finish training!
[ 2022-06-16 11:01:26,946 ] 

```




## 6 模型测试

在终端输入如下命令行进行训练：
```
python main.py -c <config> -e
```
在本次复现项目中，针对2001,2002两个model

2001，输入


```
export CUDA_VISIBLE_DEVICES=0
python ./PaddleVideo-develop/paddlevideo/tasks/main.py --gpus 0 -c 2001 -e
```

注意，输入以上命令后需要选择测试的模型，作者训练好的达标模型标注为1号，输入数字1+回车即可

结果如下所示：
```
[ 2022-06-21 06:26:31,804 ] Saving folder path: /home/aistudio/EfficientGCN_paddle-main/workdir_pad/temp
[ 2022-06-21 06:26:31,804 ] 
[ 2022-06-21 06:26:31,804 ] Starting preparing ...
[ 2022-06-21 06:26:31,804 ] Saving model name: 2001_EfficientGCN-B0_ntu-xsub
[ 2022-06-21 06:26:31,813 ] GPU-0 used: 2.75MB
[ 2022-06-21 06:26:31,832 ] Dataset: ntu-xsub
[ 2022-06-21 06:26:31,832 ] Batch size: train-16, eval-16
[ 2022-06-21 06:26:31,832 ] Data shape (branch, channel, frame, joint, person): [3, 6, 288, 25, 2]
[ 2022-06-21 06:26:31,832 ] Number of action classes: 60
W0621 06:26:33.042603  4339 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0621 06:26:33.047102  4339 device_context.cc:465] device: 0, cuDNN Version: 7.6.
[ 2022-06-21 06:26:34,944 ] Model: EfficientGCN-B0 {'stem_channel': 64, 'block_args': [[48, 1, 0.5], [24, 1, 0.5], [64, 2, 1], [128, 2, 1]], 'fusion_stage': 2, 'act_type': 'swish', 'att_type': 'stja', 'layer_type': 'SG', 'drop_prob': 0.25, 'kernel_size': [5, 2], 'scale_args': [1.2, 1.35], 'expand_ratio': 0, 'reduct_ratio': 2, 'bias': True, 'edge': True}
[ 2022-06-21 06:26:34,967 ] Pretrained model: /home/aistudio/EfficientGCN_paddle-main/pretrained/2001_EfficientGCN-B0_ntu-xsub.pdparams.tar
[ 2022-06-21 06:26:34,968 ] LR_Scheduler: cosine {'max_epoch': 70, 'warm_up': 10}
[ 2022-06-21 06:26:34,969 ] Optimizer: SGD {'momentum': 0.9, 'weight_decay': 0.0001, 'learning_rate': <paddle.optimizer.lr.LambdaDecay object at 0x7eff7f5d3cd0>, 'use_nesterov': True}
[ 2022-06-21 06:26:34,969 ] Loss function: CrossEntropyLoss
[ 2022-06-21 06:26:34,969 ] Successful!
[ 2022-06-21 06:26:34,969 ] 
[ 2022-06-21 06:26:34,969 ] Loading evaluating model ...
/home/aistudio/EfficientGCN_paddle-main/workdir_pad/2001_EfficientGCN-B0_ntu-xsub/2022-06-13 09-55-23/reco_results.json
[ 2022-06-21 06:26:34,970 ] Please choose the evaluating model from the following models.
[ 2022-06-21 06:26:34,970 ] Default is the initial or pretrained model.
[ 2022-06-21 06:26:34,970 ] (1) accuracy: 90.20% | training time: 2022-06-13 09-55-23
[ 2022-06-21 06:26:34,970 ] Your choice (number of the model, q for quit): 
Text(value='')
/home/aistudio/EfficientGCN_paddle-main/workdir_pad/2001_EfficientGCN-B0_ntu-xsub/2022-06-13 09-55-23/2001_EfficientGCN-B0_ntu-xsub.pth.tar
[ 2022-06-21 06:26:35,063 ] Successful!
[ 2022-06-21 06:26:35,063 ] 
[ 2022-06-21 06:26:35,063 ] Starting evaluating ...
100%|███████████████████████████████████████| 1030/1030 [02:51<00:00,  6.00it/s]
[ 2022-06-21 06:29:26,632 ] Top-1 accuracy: 14865/16480(90.20%), Top-5 accuracy: 16209/16480(98.36%), Mean loss:0.3799
[ 2022-06-21 06:29:26,632 ] Evaluating time: 171.57s, Speed: 96.06 sequnces/(second*GPU)
[ 2022-06-21 06:29:26,632 ] 
[ 2022-06-21 06:29:26,640 ] Finish evaluating!
```


x-view(2002)

```
export CUDA_VISIBLE_DEVICES=0
python ./PaddleVideo-develop/paddlevideo/tasks/main.py --gpus 0 -c 2002 -e
```

同理，输入以上命令后需要选择测试的模型，作者训练好的达标模型标注为1号，输入数字1+回车即可
部分测试输出如下：

```
[ 2022-06-21 06:39:35,046 ] Saving folder path: /home/aistudio/EfficientGCN_paddle-main/workdir_pad/temp
[ 2022-06-21 06:39:35,046 ] 
[ 2022-06-21 06:39:35,046 ] Starting preparing ...
[ 2022-06-21 06:39:35,046 ] Saving model name: 2002_EfficientGCN-B0_ntu-xview
[ 2022-06-21 06:39:35,055 ] GPU-0 used: 2.75MB
[ 2022-06-21 06:39:35,072 ] Dataset: ntu-xview
[ 2022-06-21 06:39:35,074 ] Batch size: train-16, eval-16
[ 2022-06-21 06:39:35,074 ] Data shape (branch, channel, frame, joint, person): [3, 6, 288, 25, 2]
[ 2022-06-21 06:39:35,074 ] Number of action classes: 60
W0621 06:39:36.046432  6296 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0621 06:39:36.050431  6296 device_context.cc:465] device: 0, cuDNN Version: 7.6.
[ 2022-06-21 06:39:37,784 ] Model: EfficientGCN-B0 {'stem_channel': 64, 'block_args': [[48, 1, 0.5], [24, 1, 0.5], [64, 2, 1], [128, 2, 1]], 'fusion_stage': 2, 'act_type': 'swish', 'att_type': 'stja', 'layer_type': 'SG', 'drop_prob': 0.25, 'kernel_size': [5, 2], 'scale_args': [1.2, 1.35], 'expand_ratio': 0, 'reduct_ratio': 2, 'bias': True, 'edge': True}
[ 2022-06-21 06:39:37,806 ] Pretrained model: /home/aistudio/EfficientGCN_paddle-main/pretrained/2002_EfficientGCN-B0_ntu-xview.pdparams.tar
[ 2022-06-21 06:39:37,807 ] LR_Scheduler: cosine {'max_epoch': 70, 'warm_up': 10}
[ 2022-06-21 06:39:37,808 ] Optimizer: SGD {'momentum': 0.9, 'weight_decay': 0.0001, 'learning_rate': <paddle.optimizer.lr.LambdaDecay object at 0x7fc06d7c8c50>, 'use_nesterov': True}
[ 2022-06-21 06:39:37,808 ] Loss function: CrossEntropyLoss
[ 2022-06-21 06:39:37,808 ] Successful!
[ 2022-06-21 06:39:37,808 ] 
[ 2022-06-21 06:39:37,808 ] Loading evaluating model ...
/home/aistudio/EfficientGCN_paddle-main/workdir_pad/2002_EfficientGCN-B0_ntu-xview/2022-06-15 12-35-17/reco_results.json
[ 2022-06-21 06:39:37,808 ] Please choose the evaluating model from the following models.
[ 2022-06-21 06:39:37,808 ] Default is the initial or pretrained model.
[ 2022-06-21 06:39:37,808 ] (1) accuracy: 94.99% | training time: 2022-06-15 12-35-17
[ 2022-06-21 06:39:37,808 ] Your choice (number of the model, q for quit): 
[ 2022-06-21 06:39:37,808 ] 1
/home/aistudio/EfficientGCN_paddle-main/workdir_pad/2002_EfficientGCN-B0_ntu-xview/2022-06-15 12-35-17/2002_EfficientGCN-B0_ntu-xview.pth.tar
[ 2022-06-21 06:39:37,871 ] Successful!
[ 2022-06-21 06:39:37,871 ] 
[ 2022-06-21 06:39:37,871 ] Starting evaluating ...
100%|███████████████████████████████████████| 1183/1183 [03:03<00:00,  6.43it/s]
[ 2022-06-21 06:42:41,740 ] Top-1 accuracy: 17980/18928(94.99%), Top-5 accuracy: 18794/18928(99.29%), Mean loss:0.1813
[ 2022-06-21 06:42:41,740 ] Evaluating time: 183.87s, Speed: 102.94 sequnces/(second*GPU)
[ 2022-06-21 06:42:41,741 ] 
[ 2022-06-21 06:42:41,749 ] Finish evaluating!

```
