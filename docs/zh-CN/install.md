简体中文 | [English](../en/install.md)

# 安装说明

---

- [简介](#简介)
- [安装PaddlePaddle](#安装PaddlePaddle)
- [安装PaddleVideo](#安装PaddleVideo)

## 简介

使用PaddleVideo之前，请先安装PaddlePaddle及相关依赖项。


## 安装PaddlePaddle

运行PaddleVideo需要`PaddlePaddle 2.0`或更高版本。请参照[安装文档](http://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。
PaddleVideo只支持python3.7及以上的运行环境，依赖项请安装python3.7及以上的安装包

如果已经安装好了cuda、cudnn、nccl或者安装好了nvidia-docker运行环境，可以pip3安装最新GPU版本PaddlePaddle

```bash
pip3 install paddlepaddle-gpu --upgrade
```

也可以从源码编译安装PaddlePaddle，请参照[安装文档](http://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

使用以下命令可以验证PaddlePaddle是否安装成功。

```python3
import paddle
paddle.utils.run_check()
```

查看PaddlePaddle版本的命令如下：

```bash
python3 -c "import paddle; print(paddle.__version__)"
```

注意：
- 从源码编译的PaddlePaddle版本号为0.0.0，请确保使用了PaddlePaddle 2.0及之后的源码编译。
- PaddleVideo基于PaddlePaddle高性能的分布式训练能力，若您从源码编译，请确保打开编译选项，**WITH_DISTRIBUTE=ON**。具体编译选项参考[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#id3)。
- 在docker中运行时，为保证docker容器有足够的共享内存用于Paddle的数据读取加速，在创建docker容器时，请设置参数`--shm_size=32g`，条件允许的话可以设置为更大的值。

**运行环境需求:**

- Python3.7 or later version (当前只支持Linux系统)
- CUDA >= 10.1
- cuDNN >= 7.6.4
- nccl >= 2.1.2


## 安装PaddleVideo

**克隆PaddleVideo模型库：**

```
cd path_to_clone_PaddleVideo
git clone https://github.com/PaddlePaddle/PaddleVideo.git
cd PaddleVideo
```

**安装Python依赖库：**

Python依赖库在[requirements.txt](https://github.com/PaddlePaddle/PaddleVideo/blob/master/requirements.txt)中给出，可通过如下命令安装：

```
python3.7 -m pip install --upgrade pip
pip3.7 install --upgrade -r requirements.txt
```

**从python安装包安装PaddleVideo：**

使用pypi安装

```bash
pip3.7 install ppvideo==2.3.0
```

安装完成后，可以使用命令行方式启动程序
```bash
ppvideo --model_name='ppTSM_v2' --use_gpu=False --video_file='data/example.avi'
```

wheel包更多的使用教程可以参考[快速开始](./quick_start.md)

---
