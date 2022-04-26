
## Slim功能介绍
复杂的模型有利于提高模型的性能，但也导致模型中存在一定冗余。此部分提供精简模型的功能，包括两部分：模型量化（量化训练、离线量化）、模型剪枝。

其中模型量化将全精度缩减到定点数减少这种冗余，达到减少模型计算复杂度，提高模型推理性能的目的。
模型量化可以在基本不损失模型的精度的情况下，将FP32精度的模型参数转换为Int8精度，减小模型参数大小并加速计算，使用量化后的模型在移动端等部署时更具备速度优势。

模型剪枝将CNN中不重要的卷积核裁剪掉，减少模型参数量，从而降低模型计算复杂度。

本教程将介绍如何使用飞桨模型压缩库PaddleSlim做PaddleVideo模型的压缩。
[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) 集成了模型剪枝、量化（包括量化训练和离线量化）、蒸馏和神经网络搜索等多种业界常用且领先的模型压缩功能，如果您感兴趣，可以关注并了解。

在开始本教程之前，建议先了解[PaddleVideo模型的训练方法](../../docs/zh-CN/usage.md)以及[PaddleSlim](https://paddleslim.readthedocs.io/zh_CN/latest/index.html)


## 快速开始
当训练出一个模型后，如果希望进一步的压缩模型大小并加速预测，可使用量化或者剪枝的方法压缩模型。

模型压缩主要包括五个步骤：
1. 安装 PaddleSlim
2. 准备训练好的模型
3. 模型压缩
4. 导出量化推理模型
5. 量化模型预测部署

### 1. 安装PaddleSlim

* 可以通过pip install的方式进行安装。

```bash
python3.7 -m pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```

* 如果获取PaddleSlim的最新特性，可以从源码安装。

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd Paddleslim
python3.7 setup.py install
```

### 2. 准备训练好的模型

PaddleVideo提供了一系列训练好的[模型](../../docs/zh-CN/model_zoo/README.md)，如果待量化的模型不在列表中，需要按照[常规训练](../../docs/zh-CN/usage.md)方法得到训练好的模型。

### 3. 模型压缩

进入PaddleVideo根目录

```bash
cd PaddleVideo
```

离线量化代码位于`deploy/slim/quant_post_static.py`。

#### 3.1 模型量化

量化训练包括离线量化训练和在线量化训练(TODO)，在线量化训练效果更好，需加载预训练模型，在定义好量化策略后即可对模型进行量化。

##### 3.1.1 在线量化训练
TODO

##### 3.1.2 离线量化

**注意**：目前离线量化，必须使用已经训练好的模型导出的`inference model`进行量化。一般模型导出`inference model`可参考[教程](../../docs/zh-CN/usage.md#5-模型推理).

一般来说，离线量化损失模型精度较多。

以PP-TSM模型为例，生成`inference model`后，离线量化运行方式如下

```bash
# 下载并解压出少量数据用于离线量化的校准
pushd ./data/k400
wget -nc https://videotag.bj.bcebos.com/Data/k400_rawframes_small.tar
tar -xf k400_rawframes_small.tar
popd

# 然后进入deploy/slim目录下
cd deploy/slim

# 执行离线量化命令
python3.7 quant_post_static.py \
-c ../../configs/recognition/pptsm/pptsm_k400_frames_uniform_quantization.yaml \
--use_gpu=True
```

除`use_gpu`外，所有的量化环境参数都在`pptsm_k400_frames_uniform_quantization.yaml`文件中进行配置
其中`inference_model_dir`表示上一步导出的`inference model`目录路径，`quant_output_dir`表示量化模型的输出目录路径

执行成功后，在`quant_output_dir`的目录下生成了`__model__`文件和`__params__`文件，这二者用于存储生成的离线量化模型
类似`inference model`的使用方法，接下来可以直接用这两个文件进行预测部署，无需再重新导出模型。

```bash
# 使用PP-TSM离线量化模型进行预测
# 回到PaddleVideo目录下
cd ../../

# 使用量化模型进行预测
python3.7 tools/predict.py \
--input_file data/example.avi \
--config configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml \
--model_file ./inference/ppTSM/quant_model/__model__ \
--params_file ./inference/ppTSM/quant_model/__params__ \
--use_gpu=True \
--use_tensorrt=False
```

输出如下：
```bash
Current video file: data/example.avi
        top-1 class: 5
        top-1 score: 0.9997928738594055
```
#### 3.2 模型剪枝
TODO


### 4. 导出模型
TODO


### 5. 模型部署

上述步骤导出的模型可以通过PaddleLite的opt模型转换工具完成模型转换。
模型部署的可参考
[Serving Python部署](../python_serving/readme.md)
[Serving C++部署](../cpp_serving/readme.md)


## 训练超参数建议

* 量化训练时，建议加载常规训练得到的预训练模型，加速量化训练收敛。
* 量化训练时，建议初始学习率修改为常规训练的`1/20~1/10`，同时将训练epoch数修改为常规训练的`1/5~1/2`，学习率策略方面，加上Warmup，其他配置信息不建议修改。
