#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle.nn import Conv2D, ReLU, AdaptiveAvgPool2D
from paddle import nn
from paddle.framework import ParamAttr
from paddle.nn.initializer import Uniform
import math
from ..registry import BACKBONES

#定义参数初始化规则
weight_ins = paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(
    0., 0.02),
                              trainable=True)
bias_ins = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0),
                            trainable=True)


@BACKBONES.register()
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    bias_ins = ParamAttr(
        initializer=Uniform(-1 / math.sqrt(in_channels * (kernel_size**1)), 1 /
                            math.sqrt(in_channels * (kernel_size**2))))
    return Conv2D(in_channels,
                  out_channels,
                  kernel_size,
                  padding=(kernel_size // 2),
                  weight_attr=weight_ins,
                  bias_attr=bias_ins)


@BACKBONES.register()
class PALayer(nn.Layer):  #像素注意力块

    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            Conv2D(channel,
                   channel // 8,
                   1,
                   padding=0,
                   weight_attr=weight_ins,
                   bias_attr=ParamAttr(
                       initializer=Uniform(-1 / math.sqrt(channel * (1**2)), 1 /
                                           math.sqrt(channel * (1**2))))),
            nn.ReLU(),
            nn.Conv2D(
                channel // 8,
                1,
                1,
                padding=0,
                weight_attr=weight_ins,
                bias_attr=ParamAttr(
                    initializer=Uniform(-1 / math.sqrt(channel // 8 *
                                                       (1**2)), 1 /
                                        math.sqrt(channel // 8 * (1**2))))),
            nn.Sigmoid())

    def forward(self, x):
        y = self.pa(x)
        return x * y


@BACKBONES.register()
class CALayer(nn.Layer):  #通道注意力块

    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.ca = nn.Sequential(
            Conv2D(channel,
                   channel // 8,
                   1,
                   padding=0,
                   weight_attr=weight_ins,
                   bias_attr=ParamAttr(
                       initializer=Uniform(-1 / math.sqrt(channel * (1**2)), 1 /
                                           math.sqrt(channel * (1**2))))),
            nn.ReLU(),
            nn.Conv2D(
                channel // 8,
                channel,
                1,
                padding=0,
                weight_attr=weight_ins,
                bias_attr=ParamAttr(
                    initializer=Uniform(-1 / math.sqrt(channel // 8 *
                                                       (1**2)), 1 /
                                        math.sqrt(channel // 8 * (1**2))))),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


@BACKBONES.register()
class Block(nn.Layer):

    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size)
        self.act1 = ReLU()
        self.conv2 = conv(dim, dim, kernel_size)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


@BACKBONES.register()
class Group(nn.Layer):

    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = paddle.nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


@BACKBONES.register()
class FFA(nn.Layer):

    def __init__(self, gps, blocks, conv=default_conv):
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2D(1),
            Conv2D(self.dim * self.gps,
                   self.dim // 16,
                   1,
                   padding=0,
                   weight_attr=weight_ins,
                   bias_attr=ParamAttr(
                       initializer=Uniform(-1 /
                                           math.sqrt(self.dim * self.gps), 1 /
                                           math.sqrt(self.dim * self.gps)))),
            nn.ReLU(),
            Conv2D(self.dim // 16,
                   self.dim * self.gps,
                   1,
                   padding=0,
                   weight_attr=weight_ins,
                   bias_attr=ParamAttr(
                       initializer=Uniform(-1 / math.sqrt(self.dim // 16), 1 /
                                           math.sqrt(self.dim // 16)))),
            nn.Sigmoid()
        ])
        self.palayer = PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)
        ]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(paddle.concat([res1, res2, res3], axis=1))
        w = paddle.reshape(w, [-1, self.gps, self.dim, 1, 1])
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out = self.palayer(out)
        x = self.post(out)
        return x + x1
