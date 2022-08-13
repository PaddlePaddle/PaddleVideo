import paddle
from paddle import nn
import paddle.nn.functional as F

class Swish(nn.Layer):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x = x.multiply(F.sigmoid(x))
            return x
        else:
            return x.multiply(F.sigmoid(x))

class HardSwish(nn.Layer):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        inner = F.relu6(x + 3.).divide(6.)
        if self.inplace:
            x = x.multiply(inner)
            return x
        else:
            return x.multiply(inner)

class AconC(nn.Layer):
    def __init__(self, channel):
        super(AconC, self).__init__()
        p1 = paddle.randn([1, channel, 1, 1], dtype="float32")

        p1 = self.create_parameter(shape=p1.shape, dtype=str(p1.numpy().dtype), attr=paddle.ParamAttr(initializer=nn.initializer.Assign(p1), trainable=True))
        self.add_parameter('p1', p1)
        p2 = paddle.randn([1, channel, 1, 1], dtype="float32")
        p2 = self.create_parameter(shape=p2.shape, dtype=str(p2.numpy().dtype), attr=paddle.ParamAttr(initializer=nn.initializer.Assign(p2), trainable=True))
        self.add_parameter('p2', p1)
        beta = paddle.ones([1, channel, 1, 1], dtype="float32")
        beta = self.create_parameter(shape=beta.shape, dtype=str(beta.numpy().dtype), attr=paddle.ParamAttr(initializer=nn.initializer.Assign(beta), trainable=True))
        self.add_parameter('beta', beta)
    def forward(self, x):
        # return (self.params[0] * x - self.params[1] * x) * F.simgoid(self.params[2] * (self.params[0] * x - self.params[1] * x)) + self.params[1] * x
        return (self.p1 * x - self.p2 * x) * F.sigmoid(self.beta * (self.p1 * x - self.p2 * x))  + self.p2 * x

class MetaAconC(nn.Layer):
    def __init__(self, channel, r=4):
        super(MetaAconC, self).__init__()
        inner_channel = max(r, channel // r)

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channel, inner_channel, 1, stride=(1, 1)),
            nn.BatchNorm2D(inner_channel),
            nn.Conv2D(inner_channel, channel, 1, stride=(1, 1)),
            nn.BatchNorm2D(channel),
            nn.Sigmoid(),
        )
        p1 = paddle.randn([1, channel, 1, 1], dtype='float32')
        p2 = paddle.randn([1, channel, 1, 1], dtype='float32')
        p1 = self.create_parameter(shape=p1.shape, dtype=str(p1.numpy().dtype), attr=paddle.ParamAttr(initializer=nn.initializer.Assign(p1), trainable=True))
        self.add_parameter('p1', p1)
        p2 = sself.create_parameter(shape=p2.shape, dtype=str(p2.numpy().dtype), attr=paddle.ParamAttr(initializer=nn.initializer.Assign(p2), trainable=True))
        self.add_parameter('p2', p2)

    def forward(self, x, **kwargs):
        return (self.p1 * x - self.p2 * x) * F.sigmoid(self.fcn(x) * (self.p1 * x - self.p2 * x)) + self.p2 * x
