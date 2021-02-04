简体中文 | [English](../../en/tutorials/accelerate.md)

- [简介](#简介)
- [模型运算加速](#模型运算加速)
- [数据读取加速](#数据读取加速)
- [训练策略加速](#训练策略加速)
- [分布式训练](#分布式训练)


# 简介

视频任务相比于图像任务的训练往往更加耗时，其原因主要有两点:
- 数据：视频解码耗时。mp4/mkv等视频文件都是经过encode后的压缩文件，通过需要经过解码和抽帧步骤才能得到原始的图像数据流，之后经过图像变换/增强操作才能将其喂入网络进行训练。如果视频帧数多，解码过程极其耗时。
- 模型：视频任务使用的模型通常有更大的参数量与计算量。为学习时序特征，视频模型一般会使用3D卷积核/(2+1)D/双流网络，这都会使得模型的参数量与计算量大大增加。

本教程介绍如下视频模型训练加速方法: 

- 模型上，通过op融合或混合精度训练的方式提升op运算效率
- 数据上，通过多进程或者并行计算的方式加速数据读取速度
- 训练策略上，通过multigrid策略减少训练耗时
- 多机分布式减少训练耗时

以上训练加速方法都已经集成进PaddleVideo中，欢迎试用~

如非特别说明，本教程所有实验的测试环境如下:
```
GPU: v100，4卡*16G
CPU: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
PaddlePaddle: 2.0.0-rc1
Cuda: 10.2
```


# 模型运算加速

- [OP融合](##OP融合)
- [混合精度训练](##混合精度训练)

## OP融合

针对[TSM模型](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/model_zoo/recognition/tsm.md)，我们实现了[temporal shift op](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fluid/layers/temporal_shift_cn.html#temporal-shift)，在节省显存的同时加速训练过程。

测试方法:
使用不同形状的Tensor，以不同的方式实现temporal shift，记录显存占用和运行时间。

测试代码:

- temporal shift op实现方式
```python
import time
import numpy as np
import paddle
import paddle.nn.functional as F

SHAPE = [32, 16, 32, 32]
#SHAPE = [128, 64, 128, 128]

otl = []
input = paddle.randn(SHAPE)
for i in range(10000):
    t1 = time.time()
    out1 = F.temporal_shift(x=input, seg_num=2, shift_ratio=0.2)
    t2 = time.time()
    ot = t2 - t1
    if i > 1000:
        otl.append(ot)
print("op time: ", sum(otl)/len(otl))
```

- 组合op实现方式
```python
import time
import numpy as np
import paddle
import paddle.nn.functional as F

SHAPE = [32, 16, 32, 32]
#SHAPE = [128, 64, 128, 128]

def temporal_shift(x, seg_num, shift_ratio):
    shape = x.shape #[N*T, C, H, W]
    reshape_x = x.reshape((-1, seg_num, shape[1], shape[2], shape[3])) #[N, T, C, H, W]
    pad_x = paddle.fluid.layers.pad(reshape_x, [0,0,1,1,0,0,0,0,0,0,]) #[N, T+2, C, H, W]
    c1 = int(shape[1] * shift_ratio)
    c2 = int(shape[1] * 2 * shift_ratio)
    slice1 = pad_x[:, :seg_num, :c1, :, :]
    slice2 = pad_x[:, 2:seg_num+2, c1:c2, :, :]
    slice3 = pad_x[:, 1:seg_num+1, c2:, :, :]
    concat_x = paddle.concat([slice1, slice2, slice3], axis=2) #[N, T, C, H, W]
    return concat_x.reshape(shape)

ctl = []
input = paddle.randn(SHAPE)
for i in range(10000):
    t2 = time.time()
    out2 = temporal_shift(x=input, seg_num=2, shift_ratio=0.2)
    t3 = time.time()
    ct = t3 - t2
    if i > 1000:
        ctl.append(ct)
print("combine time: ", sum(ctl)/len(ctl))
```

性能数据如下:

| 输入tensor形状 | 实现方式 | 显存占用/M| 计算时间/s | 加速比 |
| :------ | :-----: | :------: | :------: | :------: | 
| 32\*16\*32\*32 |op组合方式 | 1074 | 0.00029325 |  baseline |
| 32\*16\*32\*32 | temporal shift op | 1058 | 0.000045770 | **6.4x** |
| 128\*64\*128\*128 |op组合方式 | 5160 | 0.0099088 |  baseline |
| 128\*64\*128\*128 | temporal shift op | 2588 | 0.0018617 | **5.3x** |



## 混合精度训练

Comming soon~

# 数据读取加速

- [更优的解码库Decord](##更优的解码库Decord)
- [多进程加速Dataloader](##多进程加速Dataloader)
- [数据预处理DALI](##数据预处理DALI)
- [预先解码存成图像](##预先解码存成图像)

对于单机训练，视频模型的训练瓶颈大多是在数据预处理上，因此本节主要介绍在数据处理上的一些加速经验。

## 更优的解码库Decord

视频在喂入网络之前，需要经过一系列的数据预处理操作得到数据流，这些操作通常包括:

- 解码: 将视频文件解码成数据流
- 抽帧: 从视频中抽取部分帧用于网络训练
- 数据增强：缩放、裁剪、随机翻转、正则化

其中解码是最为耗时的。相较于传统的opencv或pyAV解码库，这里推荐使用性能更优的解码库[decord](https://github.com/dmlc/decord)。目前[SlowFast模型](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/model_zoo/recognition/slowfast.md)使用decord进行视频解码([源码](https://github.com/PaddlePaddle/PaddleVideo/blob/main/paddlevideo/loader/pipelines/decode_sampler.py))，对单进程的速度提升有较大作用。

我们分别以opencv/decord为解码器，实现SlowFast模型数据预处理pipeline，然后随机从kinetics-400数据集中选取200条视频，计算各pipeline处理每条视频的平均时间。

性能测试数据如下:

| 解码库 | 版本 | pipeline处理每条视频的平均时间/s | 加速比 |
| :------ | :-----: | :------: | :------: | 
| opencv | 4.2.0 | 0.20965035 | baseline |
| decord | 0.4.2 | 0.13788146 |  **1.52x** |


## 多进程加速Dataloader

数据准备好后喂入网络进行训练，网络运算使用GPU并行加速相对较快。对于单个进程来说，速度瓶颈大多在数据处理部分，GPU大部分时间是在等待CPU完成数据预处理。
飞桨2.0使用[Dataloader](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/DataLoader_cn.html#dataloader)进行数据加载，DataLoader支持单进程和多进程的数据加载方式，当 num_workers 大于0时，将使用多进程方式异步加载数据。多进程加速协作，可以overlap掉GPU大部分等待的时间，提升GPU利用率，显著加速训练过程。

我们分别设置num_workers为0或4，单卡batch_size统一设置为8，统计训练一个batch的平均耗时。

性能测试数据对比如下:
| 卡数 | 单卡num_workers | batch_cost/s | ips | 加速比 |
| :------ | :-----: | :------: |:------: |:------: |
| 单卡 | 0 | 1.763 | 4.53887 | 单卡baseline |
| 单卡 | 4 | 0.578 | 13.83729 | **3.04x** |
| 4卡 | 0 | 1.866 | 4.28733 | 多卡baseline |
| 4卡 | 4 | 0.615 | 13.00625 | **3.03x** | 

其中ips = batch_size/batch_cost，即为训练一个instance(一个video)的平均耗时。

**结合使用decord和飞桨dataloader，加上在数据增强部分做一些细节优化，SlowFast模型训练速度增益为100%，详细数据可以参考[benchmark](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/benchmark.md)**。

## 数据预处理DALI

既然GPU等待CPU进行数据处理耗时，能否把数据处理放到GPU上呢？[NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)将数据预处理pipeline转移到GPU上执行，可以显著提升训练速度。针对视频文件，DALI提供`VideoReader`op进行解码抽帧操作，但目前其仅支持连续采样的方式进行抽帧。而视频领域常用的2D模型TSN或TSM，它们均采用分段采样方式，即把视频均匀分成N段segument，然后在每个segument内随机选取一帧，最后把选取的帧组合作为输入张量。为此，我们基于DALI进行了二次开发，实现了支持分段采样方式的`VideoReader`op。为方便用户使用，我们提供了配置好的docker运行环境，具体使用方法参考[TSN-DALI使用教程](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/model_zoo/recognition/tsn_dali.md)。

测试环境: 
```
机器: Tesla v100
显存: 4卡16G
Cuda: 9.0
单卡batch_size: 32
```

性能测试数据如下:

| 加速方式  | batch耗时/s  | reader耗时/s | ips:instance/sec | 加速比 |
| :--------------- | :--------: | :------------: | :------------: | :------------: |
| DALI | 2.083 | 1.804 | 15.36597  | **1.41x** |
| Dataloader:  单卡num_workers=4 | 2.943 | 2.649 | 10.87460| baseline |
| pytorch实现 | TODO | TODO | TODO | TODO |


## 预先解码存成图像

这是一种简单直接的方法，既然视频解码耗时，那可以事先将视频解码好，存成图片，模型训练时直接读取图像即可。这种方法可以显著提升视频模型训练速度，但它也有一个很明显的缺点，就是需要耗费大量的内存空间。以kinetics-400数据集为例，共包含24万个训练样本，mp4文件约130G，解码存成图像后，占用的内存空间约为2T，所以这种方法比较适用于较小规模的数据集，如ucf-101。PaddleVideo提供了[预先解码](https://github.com/PaddlePaddle/PaddleVideo/blob/main/data/ucf101/extract_rawframes.py)的脚本，并且[TSN模型](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/model_zoo/recognition/tsn.md)和[TSM模型](https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/zh-CN/model_zoo/recognition/tsm.md)均支持直接使用frame格式的数据进行训练，详细实现参考[源码](https://github.com/PaddlePaddle/PaddleVideo/blob/main/paddlevideo/loader/dataset/frame.py)。


测试方法: 数据集选用UCF-101，模型为ppTSM，模型参数参考默认配置[pptsm.yaml](https://github.com/PaddlePaddle/PaddleVideo/blob/main/configs/recognition/tsm/pptsm.yaml)，Dataloader的num_workers参数设为0，分别以video和frame格式作为输入，单卡训练，性能数据如下:

| 数据格式  | batch耗时/s  | reader耗时/s | ips:instance/sec | reader加速比 | 加速比 |
| :--------------- | :--------: | :------------: | :------------: | :------------: | :------------: |
| frame | 1.008 | 0.591 | 15.87405  | 4.79x | **3.22x** |
| video | 3.249 | 2.832 | 4.92392| baseline | baseline |


# 训练策略加速

前述方法大多从工程的角度思考训练速度的提升，在算法策略上，FAIR在CVPR 2020中提出了[Multigrid加速策略算法](https://arxiv.org/abs/1912.00998)，它的基本思想如下: 

在图像分类任务中，若经过预处理后图像的高度和宽度分别为H和W，batch_size为N，则网络输入batch的Tensor形状为`[N, C, H, W]`，其中C等于3，指RGB三个通道。
对应到视频任务，由于增加了时序通道，输入batch的Tensor形状为`[N, C, T, H, W]`。
传统的训练策略中，每个batch的输入Tensor形状都是固定的，即都是`[N, C, T, H, W]`。若以高分辨的图像作为输入，即设置较大的`[T, H, W]`，则模型精度会高一些，但训练会更慢；若以低分辨的图像作为输入，即设置较小的`[T, H, W]`，则可以使用更大的batch size，训练更快，但模型精度会降低。在一个epoch中，能否让不同batch的输入Tensor的形状动态变化，既能提升训练速度，又能保证模型精度？

基于以上思想，FAIR在实验的基础上提出了Multigrid训练策略: 固定`N*C*T*H*W`的值，降低`T*H*W`时增大`N`的值，增大`T*H*W`时减小`N`的值。具体包含两种策略：

- Long cycle: 设完整训练需要N个epoch，将整个训练过程分4个阶段，每个阶段对应的输入tensor形状为:
```
[8N, T/4, H/sqrt(2), W/sqrt(2)], [4N, T/2, H/sqrt(2), W/sqrt(2)], [2N, T/2, H, W], [N, T, H, W]
```

- Short cycle: 在Long cycle的基础上，Short-cycle让每个iter的输入Tensor形状都会发生变化，变化策略为:
```
[H/2, W/2], [H/sqrt(2), W/sqrt(2)], [H, W]
```

# 分布式训练 

Comming soon~
