# TSN模型DALI加速训练

- [简介](#简介)
- [环境配置](#环境配置)
- [模型训练](#模型训练)

## 简介
训练速度慢是视频模型训练常见的问题，PaddleVideo使用飞桨2.0的dataloader，凭借其优异的多进程加速能力，训练模型的训练速度可以显著增加。TSN是视频领域常用的2D模型，我们使用nvidia DALI进行GPU解码，对其训练速度进行了进一步优化:
针对TSN模型，我们基于DALI进行了二次开发，实现了其均匀分段的帧采样方式。

### 性能

测试环境: Tesla v100 14卡16G，Cuda9，单卡batch_size=32。

训练速度对比如下:

| 加速方式  | batch耗时/s  | reader耗时/s | ips:instance/sec |
| :--------------- | :--------: | :------------: | :------------: |
| DALI | 2.083 | 1.804 | 15.36597  |
| Dataloader:  单卡num_workers=4 | 2.943 | 2.649 | 10.87460|
| pytorch实现 | TODO | TODO | TODO |

可以看到，使用DALI可以进一步加速模型训练，相较于dataloader，加速比约为1.5倍。

## 环境配置

我们提供docker运行环境方便您使用，docker镜像为:

```
    huangjun12/paddlevideo:tsn_dali_cuda9_0
```

基于以上docker镜像创建docker容器，运行命令为:
```
nvidia-docker run --name tsn-DALI -v /home:/workspace --network=host -it --shm-size 64g -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video huangjun12/paddlevideo:tsn_dali_cuda9_0 /bin/bash
```
- docker中安装好了飞桨2.0.0-rc1版本和我们二次开发后的DALI，创建容器后您可以在docker环境中直接开始tsn模型训练，无需额外配置环境。

## 模型训练

(1) 打开工作目录
```
cd /workspace
```

(2) 克隆PaddleVideo代码
```
git clone https://github.com/PaddlePaddle/PaddleVideo.git
```

```
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_tsn main.py --train_dali -c configs/recognition/tsn/tsn_dali.yaml -o log_level="INFO"
```









