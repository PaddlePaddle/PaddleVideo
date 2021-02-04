[简体中文](../zh-CN/install.md) | English

# Installation

---

## Introducation

This document introduces how to install PaddleVideo and its requirements.

## Install PaddlePaddle

Python 3.7, CUDA 10.1, CUDNN7.6.4 nccl2.1.2 and later version are required at first, For now, PaddleVideo only support training on the GPU device. Please follow the instructions in the [Installation](http://www.paddlepaddle.org.cn/install/quick) if the PaddlePaddle on the device is lower than v2.0

Install PaddlePaddle

```bash
pip3 install paddlepaddle-gpu --upgrade
```

or compile from source code, please refer to [Installation](http://www.paddlepaddle.org.cn/install/quick).

Verify Installation

```python
import paddle
paddle.utils.run_check()
```

Check PaddlePaddle version：

```bash
python3 -c "import paddle; print(paddle.__version__)"
```

Note:
- Make sure the compiled version is later than PaddlePaddle2.0.
- Indicate **WITH_DISTRIBUTE=ON** when compiling, Please refer to [Instruction](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#id3) for more details.
- When running in the docker, in order to ensure that the container has enough shared memory for data read acceleration of Paddle, please set the parameter `--shm_size=32g` at creating a docker container, if conditions permit, you can set it to a larger value.


## Install PaddleVideo

**Clone PaddleVideo:**

```bash
cd path_to_clone_PaddleVideo
git clone https://github.com/PaddlePaddle/PaddleVideo.git
```

**Install requirements**

```bash
pip3 install --upgrade -r requirements.txt
```

---

**Install python package**

Install PaddleVideo via pip <sup>WIP</sup>

**Install docker**

Install PaddleVideo via docker <sup>WIP</sup> 


