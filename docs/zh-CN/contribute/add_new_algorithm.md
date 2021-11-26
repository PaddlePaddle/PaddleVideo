# 添加新算法

PaddleVideo将一个算法分解为以下几个部分，并对各部分进行模块化处理，方便快速组合出新的算法。

* [1. 数据加载和处理](#1)
* [2. 网络](#2)
* [3. 优化器](#3)
* [4. 训练策略](#4)
* [5. 指标评估](#5)

示例代码如下：
```python
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
import paddle.nn as nn

# 1. 数据加载和处理
## 1.2 数据预处理Pipeline
class ExamplePipeline(object):
    """ Example Pipeline"""
    def __init__(self, mean=0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        data = results['data']
        norm_data = (data - self.mean) / self.std
        results['data'] = norm_data
        return results

## 1.1 数据集类
class ExampleDataset(Dataset):
    """ExampleDataset"""
    def __init__(self):
        super(ExampleDataset, self).__init__()
        self.x = np.random.rand(100, 20, 20)
        self.y = np.random.randint(10, size = (100, 1))

    def __getitem__(self, idx):
        x_item = self.x[idx]
        results = {}
        results['data'] = x_item
        pipeline = ExamplePipeline()
        results = pipeline(results)
        x_item = results['data'].astype('float32')
        y_item = self.y[idx].astype('int64')
        return x_item, y_item

    def __len__(self):
        return self.x.shape[0]

train_dataset = ExampleDataset()
## 1.3 封装为Dataloader对象
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# 2. 网络
class ExampleModel(nn.Layer):
    """Example Model"""
    def __init__(self):
        super(ExampleModel, self).__init__()
        ## 2.1 网络Backbobe
        self.layer1 = paddle.nn.Flatten(1, -1)
        self.layer2 = paddle.nn.Linear(400, 512)
        self.layer3 = paddle.nn.ReLU()
        self.layer4 = paddle.nn.Dropout(0.2)
        ## 2.2 网络Head
        self.layer5 = paddle.nn.Linear(512, 10)

    def forward(self, x):
        """ model forward"""
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)
        return y

model = ExampleModel()
model.train()

# 3. 优化器
optim = paddle.optimizer.Adam(parameters=model.parameters())

epochs = 5
for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader()):
        x_data = data[0]  
        y_data = data[1]  
        predicts = model(x_data)  

        ## 2.3 网络Loss
        loss = paddle.nn.functional.cross_entropy(predicts, y_data)

        acc = paddle.metric.accuracy(predicts, y_data)

        loss.backward()
        print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))

        optim.step()
        optim.clear_grad()
```
上述代码的运行输出日志如下：
```txt
epoch: 0, batch_id: 0, loss is: [2.5613842], acc is: [0.]
epoch: 0, batch_id: 1, loss is: [2.5776138], acc is: [0.1]
epoch: 0, batch_id: 2, loss is: [2.551022], acc is: [0.1]
epoch: 0, batch_id: 3, loss is: [2.782001], acc is: [0.]
epoch: 0, batch_id: 4, loss is: [2.787499], acc is: [0.1]
```
将以上代码集成进PaddleVideo的示例pr参考 [#257](https://github.com/PaddlePaddle/PaddleVideo/pull/257)

下面将分别对每个部分进行介绍，并介绍如何在该部分里添加新算法所需模块。

<a name="1"></a>

## 1. 数据加载和处理

数据加载和处理部分由`Dataset类`、`预处理Pipeline`和`Dataloader对象`组成。`Dataset类`是数据集类，其中的`__getitem__`方法定义了每一个视频样本数据的处理方式。`预处理Pipeline`定义了数据预处理步骤，包括视频的读取，解码以及数据增强等操作。`预处理定义的Pipeline`通常在`Dataset类`的`__getitem__`方法中被调用，以完成对视频预处理操作。这一部分在[paddlevideo/loader](../../../paddlevideo/loader)下。 各个文件及文件夹作用说明如下:

```txt
paddlevideo/loader/
├── dataset
│   ├── base.py            # Dataset基类
│   ├── frame.py           # 处理Frame格式输入的Dataset类
│   └── video.py           # 处理Video格式输入的Dataset类
├── pipelines
│   ├── decode.py          # 解码Pipeline，对视频进行解码
│   ├── sample.py          # 抽帧Pipeline，对视频抽帧的方式
│   ├── augmentations.py   # 数据增强Pipeline，包括缩放、裁剪、反转、正则化等
...
```

PaddleVideo内置了针对不同数据集的Dataset相关模块，对于没有内置的模块可通过如下步骤添加:

1. 在 [paddlevideo/loader/dataset](../../../paddlevideo/loader/dataset) 文件夹下新建文件，如my_dataset.py。
2. 在 my_dataset.py 文件内添加相关代码，示例代码如下:

```python
@DATASETS.register()  # 通过装饰器，自动进行注册
class MyDataset:
    def __init__(self, *args, **kwargs):
        # your init code
        pass

    def load_file(self):
        info = []
        # load file list
        return info

    def prepare_train(self, idx):
        results = copy.deepcopy(self.info[idx])
        results = self.pipeline(results) # train pipeline  
        return results['image'], results['labels'] #return your data item

    def prepare_test(self, idx):
        results = copy.deepcopy(self.info[idx])
        results = self.pipeline(results) # test pipeline  
        return results['image'], results['labels'] #return your data item
```

3. 在 [paddlevideo/loader/dataset/\_\_init\_\_.py](../../../paddlevideo/loader/dataset/__init__.py) 文件内导入添加的模块。

最后在config文件中指定Dataset类名即可使用。如:

```yaml
# Define your Dataset name and args
DATASET:
    batch_size: 16                                # single-card bacth size
    num_workers: 4                                # the number of subprocess on each GPU.
    train:
        format: "FrameDataset"                    # Dataset class
        data_prefix: "data/k400/rawframes"        # train data root path
        file_path: "data/k400/train_frames.list"  # train data list file path
        suffix: 'img_{:05}.jpg'
    valid:
        format: "FrameDataset"                    # Dataset class
        data_prefix: "data/k400/rawframes"        # valid data root path
        file_path: "data/k400/train_frames.list"  # valid data list file path
        suffix: 'img_{:05}.jpg'
    test:
        format: "FrameDataset"                    # Dataset class
        data_prefix: "data/k400/rawframes"        # test data root path
        file_path: "data/k400/train_frames.list"  # test data list file path
        suffix: 'img_{:05}.jpg'
```

- 关于模块注册机制的详细说明，可以参考[配置系统设计](./config.md)

PaddleVideo内置了大量视频编解码及图像变换相关模块，对于没有内置的模块可通过如下步骤添加:

1. 在 [paddlevideo/loader/pipelines](../../../paddlevideo/loader/pipelines) 文件夹下新建文件，如my_pipeline.py。
2. 在 my_pipeline.py 文件内添加相关代码，示例代码如下:

```python
@PIPELINES.register()  # 通过装饰器，自动进行注册
class MyPipeline:  
    def __init__(self, *args, **kwargs):
        # your init code
        pass

    def __call__(self, results):
        img = results['image']
        label = results['label']
        # your process code

        results['image'] = img
        results['label'] = label
        return results
```

3. 在 [paddlevideo/loader/pipelines/\_\_init\_\_.py](../../../paddlevideo/loader/pipelines/__init__.py) 文件内导入添加的模块。

数据处理的所有处理步骤由不同的模块顺序执行而成，在config文件中按照列表的形式组合并执行。如:

```yaml
# Define your pipeline name and args
PIPELINE:
    train:
        decode:
            name: "FrameDecoder"             # Pipeline Class name
        sample:
            name: "Sampler"                  # Pipeline Class name
            num_seg: 8                       # init args
            seg_len: 1                       # init args
            valid_mode: False                # init args
        transform:
            - Scale:                         # Pipeline Class name
                short_size: 256              # init args
```

<a name="2"></a>

## 2. 网络

网络部分完成了网络的组网操作，PaddleVideo将网络划分为四三部分，这一部分在[paddlevideo/modeling](../../../paddlevideo/modeling)下。 进入网络的数据将按照顺序(backbones->heads->loss)依次通过这三个部分。backbone用于特征提取，loss通过heads的[loss方法](https://github.com/PaddlePaddle/PaddleVideo/blob/5f7e22f406d11912eef511bafae28c594ccaa07e/paddlevideo/modeling/heads/base.py#L67)被调用。除了损失值，训练过程中如果想观察其它的精度指标(如top1, top5)，也可以在head中定义相应的计算方法，参考[get_acc方法](https://github.com/PaddlePaddle/PaddleVideo/blob/5f7e22f406d11912eef511bafae28c594ccaa07e/paddlevideo/modeling/heads/base.py#L122)，loss模块最终返回一个[loss字典](https://github.com/PaddlePaddle/PaddleVideo/blob/5f7e22f406d11912eef511bafae28c594ccaa07e/paddlevideo/modeling/heads/base.py#L81)，存储loss值以及其它需要的精度指标。

```bash
├── framework     # 组合backbones->heads->loss，定义从输入数据到输出loss的过程
├── backbones     # 网络的特征提取模块
├── heads         # 网络的输出模块
└── losses        # 网络的损失函数模块
```

PaddleVideo内置了TSN、TSM、SlowFast、ST-GCN、BMN等算法相关的常用模块，对于没有内置的模块可通过如下步骤添加，四个部分添加步骤一致，以backbones为例:

1. 在 [paddlevideo/modeling/backbones](../../../paddlevideo/modeling/backbones) 文件夹下新建文件，如my_backbone.py。
2. 在 my_backbone.py 文件内添加相关代码，示例代码如下:

```python
@BACKBONES.register()    # 通过装饰器，自动进行注册
class MyBackbone(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(MyBackbone, self).__init__()
        # your init code
        self.conv = nn.xxxx

    def forward(self, inputs):
        # your network forward
        y = self.conv(inputs)
        return y
```

3. 在 [paddlevideo/modeling/backbones/\_\_init\_\_.py](../../../paddlevideo/modeling/backbones/__init__.py)文件内导入添加的模块。

在完成网络的四部分模块添加之后，只需要配置文件中进行配置即可使用，如:

```yaml
MODEL:
    framework: "Recognizer2D"    # Framework class name
    backbone:  
        name: "ResNetTweaksTSM"  # Backbone class name
        depth: 50                # init args
    head:
        name: "ppTSMHead"        # Heads class name
        num_classes: 400         # init args
    loss:
        name: "MyLoss"           # Losses class name
        scale: 0.1               # init args
```

<a name="3"></a>

## 3. 优化器

优化器用于训练网络。优化器内部还包含了网络正则化和学习率衰减模块。 这一部分在[paddlevideo/solver/](../../../paddlevideo/solver/)下。 PaddleVideo内置了飞桨框架所有的[优化器模块](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.1/api/paddle/optimizer/Overview_cn.html#api)和[学习率衰减模块](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.1/api/paddle/optimizer/Overview_cn.html#about-lr)。只需要在配置文件中指定相应模块名称及参数即可方便的调用，示例：

```yaml
OPTIMIZER:
    name: 'Momentum'                        # Optimizer class name
    momentum: 0.9                           # init args
    learning_rate:  
        name: 'PiecewiseDecay'              # Learning rate scheduler class name
        boundaries: [10, 20]                # init args
        values: [0.001, 0.0001, 0.00001]    # init args
```

对于没有内置的模块可通过如下步骤添加，以`learning rate`为例:

1. 在 [paddlevideo/solver/custom_lr.py](../../../paddlevideo/solver/custom_lr.py) 文件内创建自己的学习率调整策略，示例代码如下:

```python
class MyLR(LRScheduler):
    def __init__(self, *args, **kwargs):
        self.learning_rate = learning_rate

    def step(self, epoch):
        # learning rate step scheduler
        self.last_lr = xxx

```

在学习率模块添加之后，只需要配置文件中进行配置即可使用，如:

```yaml
OPTIMIZER:
  name: 'Momentum'
  momentum: 0.9
  learning_rate:
    iter_step: True
    name: 'CustomWarmupCosineDecay'   # LR class name
    max_epoch: 80                     # init args
    warmup_epochs: 10                 # init args
```

<a name="4"></a>

## 4. 训练策略

PaddleVideo内置了很多模型训练相关trick，包括标签平滑、数据增强Mix-up、PreciseBN等，只需要在配置文件中指定相应模块名称及参数即可方便的调用，示例：

```yaml

MODEL:
    framework: "Recognizer2D"
    backbone:
        name: "ResNetTweaksTSM"
    head:
        name: "ppTSMHead"
        ls_eps: 0.1                  # ls_eps字段添加label smooth，并指定平滑系数

MIX:
    name: "Mixup"                    # 添加数据增强 Mix-up策略
    alpha: 0.2                       # 指定mix系数

PRECISEBN:                           # 添加preciseBN策略
  preciseBN_interval: 5              # 指定prciseBN间隔
  num_iters_preciseBN: 200           # 指定preciseBN运行的batchs数量

```

训练相关的代码通过[paddlevideo/tasks/train.py](../../../paddlevideo/tasks/train.py)被组织起来，最终被[PaddleVideo/main.py](../../../../PaddleVideo/main.py)调用启动训练，单卡训练和多卡训练的启动方式略有不同。单卡训练启动方式如下:

```bash
export CUDA_VISIBLE_DEVICES=0         #指定使用的GPU显卡id
python3.7 main.py  --validate -c configs_path/your_config.yaml
```
- `--validate` 参数指定训练时运行validation
- `-c` 参数指定配置文件路径
-  `-o`: 指定重写参数，例如： `-o DATASET.batch_size=16` 用于重写train时batch size大小

多卡训练通过paddle.distributed.launch启动，方式如下:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7 -B -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7"  --log_dir=log_pptsm  main.py  --validate -c configs/recognition/pptsm/pptsm_k400_frames_uniform.yaml
```
- `--gpus`参数指定使用的GPU显卡id
- `--log_dir`参数指定日志保存目录
多卡训练详细说明可以参考[单机多卡训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.1/guides/02_paddle2.0_develop/06_device_cn.html#danjiduokaxunlian)


<a name="5"></a>

## 5. 指标评估

训练完成后，需要进行指标评估，paddlevideo将指标评估模块与训练模块解耦，通过在[PaddleVideo/main.py](../../../../PaddleVideo/main.py)运行时指定`--test`参数调用test模块进行指标评估，评估方法的实现主体在[paddlevideo/metrics/](../../../paddlevideo/metrics)下。 PaddleVideo内置了Uniform、Dense等相关的指标评估模块，对于没有内置的模块可通过如下步骤添加:

1. 在 [paddlevideo/metrics/](../../../paddlevideo/metrics/) 文件夹下新建文件，如my_metric.py。
2. 在 my_metric.py 文件内添加相关代码，示例代码如下:

```python
@METRIC.register        # 通过装饰器，自动进行注册
class MyMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        self.top1 = []

    def update(self, batch_id, data, outputs):
        # update metrics during each iter
        self.top1.append(xx)

    def accumulate(self):
        # accumulate metrics when finished all iters.
        xxx
        print(np.mean(np.array(self.top1)))

```

3. 在 [paddlevideo/metrics/\_\_init\_\_.py](../../../paddlevideo/metrics/__init__.py)文件内导入添加的模块。

在指标评估模块添加之后，只需要配置文件中进行配置即可使用，如:

```yaml
METRIC:
    name: 'CenterCropMetric'    # Metric class name
```

模型测试运行方法如下：
```bash
python3.7 main.py --test -c config_path/your_config.yaml -w weight_path/your_weight.pdparams
```
- `--test`参数指定运行测试模式
- `-c`参数指定配置文件
- `-w`参数指定训练好的权重保存路径

