# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 filter_size,
                 stride,
                 padding,
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = nn.Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            weight_attr=ParamAttr(name=name + ".conv.weights"),
            bias_attr=False)

        bn_name = name + ".bn"
        self._bn = nn.BatchNorm(
            num_channels=output_channels,
            act="leaky_relu",
            param_attr=ParamAttr(name=bn_name + ".scale"),
            bias_attr=ParamAttr(name=bn_name + ".offset"),
            moving_mean_name=bn_name + ".mean",
            moving_variance_name=bn_name + ".var")

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._bn(x)
        return x


class BasicBlock(nn.Layer):
    def __init__(self, input_channels, output_channels, name=None):
        super(BasicBlock, self).__init__()

        self._conv1 = ConvBNLayer(input_channels=input_channels, output_channels=output_channels, filter_size=[
                                  3, 3], stride=1, padding=1,  name=name+'.0')
        self._max_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self._conv2 = ConvBNLayer(input_channels=output_channels, output_channels=output_channels *
                                  2, filter_size=[3, 3], stride=1, padding=1, name=name+'.1')
        self._conv3 = ConvBNLayer(input_channels=output_channels*2, output_channels=output_channels,
                                  filter_size=[1, 1], stride=1, padding=0, name=name+'.2')

    def forward(self, x):
        x = self._conv1(x)
        x = self._max_pool(x)
        x = self._conv2(x)
        x = self._conv3(x)
        return x


class Reorg(nn.Layer):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.dim() == 4)
        B = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.reshape([B, C, H // hs, hs, W // ws, ws]
                      ).transpose([0, 1, 2, 4, 3, 5])
        x = x.reshape([B, C, H // hs * W // ws, hs * ws]
                      ).transpose([0, 1, 3, 2])
        x = x.reshape([B, C, hs * ws, H // hs, W // ws]
                      ).transpose([0, 2, 1, 3, 4])
        x = x.reshape([B, hs * ws * C, H // hs, W // ws])
        return x


class Darknet(nn.Layer):
    def __init__(self, pretrained=None):
        super(Darknet, self).__init__()
        self.pretrained = pretrained
        self._conv1 = ConvBNLayer(
            input_channels=3, output_channels=32, filter_size=3, stride=1, padding=1, name='input')
        self._max_pool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self._basic_block_11 = BasicBlock(
            input_channels=32, output_channels=64, name='1.1')
        self._basic_block_12 = BasicBlock(
            input_channels=64, output_channels=128, name='1.2')
        self._basic_block_13 = BasicBlock(
            input_channels=128, output_channels=256, name='1.3')
        self._conv2 = ConvBNLayer(
            input_channels=256, output_channels=512, filter_size=3, stride=1, padding=1, name='up1')
        self._conv3 = ConvBNLayer(
            input_channels=512, output_channels=256, filter_size=1, stride=1, padding=0, name='down1')
        self._conv4 = ConvBNLayer(
            input_channels=256, output_channels=512, filter_size=3, stride=1, padding=1, name='2.1')
        self._max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self._conv5 = ConvBNLayer(
            input_channels=512, output_channels=1024, filter_size=3, stride=1, padding=1, name='2.2')
        self._conv6 = ConvBNLayer(input_channels=1024, output_channels=512,
                                  filter_size=1, stride=1, padding=0, name='2.3')  # ori
        self._conv7 = ConvBNLayer(
            input_channels=512, output_channels=1024, filter_size=3, stride=1, padding=1, name='up2')
        self._conv8 = ConvBNLayer(input_channels=1024, output_channels=512,
                                  filter_size=1, stride=1, padding=0, name='down2')
        self._conv9 = ConvBNLayer(
            input_channels=512, output_channels=1024, filter_size=3, stride=1, padding=1, name='3.1')
        self._conv10 = ConvBNLayer(
            input_channels=1024, output_channels=1024, filter_size=3, stride=1, padding=1, name='3.2')
        self._conv11 = ConvBNLayer(
            input_channels=1024, output_channels=1024, filter_size=3, stride=1, padding=1, name='3.3')
        self._conv12 = ConvBNLayer(
            input_channels=512, output_channels=64, filter_size=1, stride=1, padding=0, name='4.1')
        self._reorg = Reorg()
        self._conv13 = ConvBNLayer(
            input_channels=1280, output_channels=1024, filter_size=3, stride=1, padding=1, name='5.1')
        self._conv14 = nn.Conv2D(1024, 425, kernel_size=1)

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._max_pool1(x)
        x = self._basic_block_11(x)
        x = self._basic_block_12(x)
        x = self._basic_block_13(x)
        x = self._conv2(x)
        x = self._conv3(x)
        ori = self._conv4(x)
        x = self._max_pool2(ori)
        x = self._conv5(x)
        x = self._conv6(x)
        x = self._conv7(x)
        x = self._conv8(x)
        x = self._conv9(x)
        x = self._conv10(x)
        x1 = self._conv11(x)
        x2 = self._conv12(ori)
        x2 = self._reorg(x2)
        x = paddle.concat([x2, x1], 1)
        x = self._conv13(x)
        x = self._conv14(x)
        return x
