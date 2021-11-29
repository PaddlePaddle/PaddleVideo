简体中文 | [English](../../en/tutorials/config.md)

# 配置系统设计

---

本文档将介绍PaddleVideo利用依赖注入技术实现控制反转，来对整个系统进行解耦，通过可自定义调整的配置文件来控制整个系统从而实现模块化。最后，介绍了配置文件和PaddleVideo运行时参数的含义。


## 设计原则

首先，模型库中会有很多对一个类实例化的操作，例如：

```python
class TSM():
    pass

model = TSM(init_attributes)
```
当越来越多的实例被创建，这种调用方法和被调用方法间的联系陡然上升，增加了整个系统的耦合性，对启用新功能建设，或是对已用功能扩展产生不便。
当然我们可以建立一个工厂模式来解决这个问题，根据配置文件的指定输入，来统一的做条件判断：

```python
if model_name == "TSM":
    model = TSM()
elif model_name == "TSN":
    model = TSN()
elif ...
```
或是像如下代码片段

```python
optimizer_cfg = dict(name:"MOMENTUM", params: XXX)
if optimizer_cfg.name = "MOMENTUM":
    optimizer = MOMENTUM(optimizer_cfg.pop(name))
elif:
    ...
```

可是，越来越多的条件判断被创建出来，还是没有统一彻底的解决这个问题。
而在其他系统中被广泛利用的 控制反转/依赖注入 技术，PaddleVideo将其利用起来进行系统解耦，并应用到诸如 LOSS METRICS BACKBONE HEAD等场景中。
PaddleVideo实现了两个组件用于完成控制反转/依赖注入：

- Register, 注册器，用于注册一个模块组件
- Builder, 用于建立（实例化）一个已注册的组件

1. Register 注册器

PaddleVideo实现了类似setter和getter方法

[source code](../../paddlevideo/utils/registry.py)

```python
#excerpt from source code.
class Registry():
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    #mapping name -> object
    def register(self,  obj, name):
        self._obj_map[name] = obj

    #get object
    def get(self, name):
        ret = self._obj_map.get(name)
        return ret
```

用于建立字符串和对象的map，如下的代码将ResNet类注册到BACKBONE map中

```python

    BACKBONES = Registry('backbone')
    class ResNet:
        pass
    BACKBONES.register(ResNet)
```

或是通过python3语法糖来装饰一个类

```python
    BACKBONES = Registry('backbone') #new a Register
    @BACKBONES.register() #regist resnet as a backbone.
    class ResNet:
        pass
```

2. Builder

应用python的反射机制，调用get方法 得到一个已经注册的模块：
```python
    # Usage: To build a module.

    backbone_name = "ResNet"
    b = BACKBONES.get(backbone_name)()
```

至此，PaddleVideo注册了一个实例，不是在他的调用地方，而是在他的声明处，一个简单的IoC系统建立起来了。
最后，PaddleVideo 通过这种方式建立了所有组件，并和配置文件参数一一对应。这里，一一对应的含义是：配置文件中的字段，`name` 代表着类的名字，其余字段对应着这个类的初始化参数。当然，除了`name` 我们也应用了别的名字来标记类名，例如：`framework`

```yaml
head:
    name: "TSMHead"  # class name
    num_classes: 400 # TSMHead class init attributes
    ...
```

---

## 配置参数

配置文件中，有多组字段，如下

- **MODEL:** 代笔模型结构
- **DATASET:** 数据集和dataloader配置
- **PIPELINE:** 数据处理流程配置字段
- **OPTIMIZER:** 优化器字段

和一些共有的参数， 如：

- model_name
- log_interval
- epochs
- resume_epoch
- log_level
...

## 模块概览

<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Architectures</b>
      </td>
      <td>
        <b>Frameworks</b>
      </td>
      <td>
        <b>Components</b>
      </td>
      <td>
        <b>Data Augmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul><li><b>Recognition</b></li>
          <ul>
            <li>TSN</li>
            <li>TSM</li>
            <li>SlowFast</li>
            <li>PP-TSM</li>
            <li>VideoTag</li>
            <li>AttentionLSTM</li>
          </ul>
        </ul>
        <ul><li><b>Localization</b></li>
          <ul>
            <li>BMN</li>
          </ul>
        </ul>
      </td>
      <td>
          <li>Recognizer1D</li>
          <li>Recognizer2D</li>
          <li>Recognizer3D</li>
          <li>Localizer</li>
        <HR></HR>
        <ul>Backbone
            <li>resnet</li>
            <li>resnet_tsm</li>
            <li>resnet_tweaks_tsm</li>
            <li>bmn</li>
        </ul>
        <ul>Head
            <li>pptsm_head</li>
            <li>tsm_head</li>
            <li>tsn_head</li>
            <li>bmn_head</li>
            <slowfast_head></li>
            <bmn_head></li>
        </ul>
      </td>
      <td>
        <ul><li><b>Solver</b></li>
          <ul><li><b>Optimizer</b></li>
              <ul>
                <li>Momentum</li>
                <li>RMSProp</li>
              </ul>
          </ul>
          <ul><li><b>LearningRate</b></li>
              <ul>
                <li>PiecewiseDecay</li>
              </ul>
          </ul>
        </ul>
        <ul><li><b>Loss</b></li>
          <ul>
            <li>CrossEntropy</li>
            <li>BMNLoss</li>  
          </ul>  
        </ul>  
        <ul><li><b>Metrics</b></li>
          <ul>
            <li>CenterCrop</li>
            <li>MultiCrop</li>  
          </ul>  
        </ul>
      </td>
      <td>
        <ul><li><b>Video</b></li>
          <ul>
            <li>Mixup</li>
            <li>Cutmix</li>  
          </ul>  
        </ul>
        <ul><li><b>Image</b></li>
            <ul>
                <li>Scale</li>
                <li>Random FLip</li>
                <li>Jitter Scale</li>  
                <li>Crop</li>
                <li>MultiCrop</li>
                <li>Center Crop</li>
                <li>MultiScaleCrop</li>
                <li>Random Crop</li>
                <li>PackOutput</li>
            </ul>
         </ul>
      </td>  
    </tr>


</td>
    </tr>
  </tbody>
</table>

---
