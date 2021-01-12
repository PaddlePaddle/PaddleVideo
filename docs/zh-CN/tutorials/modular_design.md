简体中文 | [English](../../en/tutorials/modular_design.md)

# 模型库设计

在简单了解了PaddleVideo的[配置系统](config.md)如何工作后，我们来看下PaddleVideo是怎么构建起来一个视频分类或动作定位模型的。

1. 基础知识 从一个NN layer开始

神经网络的搭建一般由多个layer组成，PaddlePaddle2.0动态图以nn.layer为构建单元，开始构建一个网络结构。
nn.Layer 的源代码请参考[nn.Layer源码](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/dygraph/layers.py#L76)
相关文档参考[nn.Layer文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fluid/dygraph/layers/Layer_cn.html#layer)

PaddleVideo主要用到的nn.Layer属性和方法有

- train() 用于设定层为训练模式
- eval() 用于设定层为预测模式
- forward() 定义每次调用执行时的计算
- state_dict() 获取层的参数
- set_state_dict() 设置层的参数

其中forward 重写了类的call方法，调用forward会执行图的计算。

2. 继承自nn.Layer的模块基类

PaddleVideo的网络结构以 nn.Layer 为构建单元，大多数网络结构模块以nn.Layer为基类进行扩展

如以下代码[head base](https://github.com/PaddlePaddle/PaddleVideo/blob/main/paddlevideo/modeling/heads/base.py#L28)

```python
class BaseHead(nn.Layer):
    def __init__():
        XXX
    def init_weights()
    
    def forward()
    
    ...
```

再比如：


3. 构建不同类型的framework

PaddleVideo 设计了不同种类的framework, 当然，他们也是继承自nn.Layer的，所有模型都对应着一种framework
包括

4. 构建framework的组件，Head，Backbone等

5. 构建数据读取器，优化器等

6. 通过配置组合各个组件

7. 正向网络

8. 反向网络

9. 指标

10. 分析实验结果



