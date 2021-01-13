# TSN模型DALI加速训练

- [简介](#简介)
- [开始使用](#开始使用)

## 简介
训练速度慢是视频模型训练的常见问题，PaddleVideo使用飞桨2.0的dataloader，凭借其优异的多进程加速能力，训练模型的训练速度可以显著增加。TSN是视频领域常用的2D模型，我们使用nvidia DALI进行GPU解码，对其训练速度进行了进一步优化:
针对TSN模型，我们基于DALI进行了二次开发，实现了其视频分段帧采样方式。

### 性能

测试环境: Tesla v100 14卡16G，Cuda9，单卡batch_size=32

| 加速方式  | batch耗时/s  | reader耗时/s | ips:instance/sec |
| :--------------- | :--------: | :------------: | :------------: |
| DALI | 2.083 | 1.804 | 15.36597  |
| Dataloader:  单卡num_workers=4 | 2.943 | 2.649 | 10.87460|
| pytorch实现 | TODO | TODO | TODO |

## 开始使用





