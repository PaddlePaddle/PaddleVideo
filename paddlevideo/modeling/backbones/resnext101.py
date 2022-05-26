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

from functools import partial
from paddle import fluid
import paddle.nn as nn
import paddle


class BatchNorm3D(nn.BatchNorm3D):
    def __init__(self, num_features, momentum=0.9, epsilon=1e-05, weight_attr=None, bias_attr=None, data_format='NCDHW',
                 name=None):
        super().__init__(
            num_features=num_features,
            momentum=momentum,
            epsilon=epsilon,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.)),
            bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.)),
            data_format=data_format,
            name=name
        )


class Conv3D(nn.Conv3D):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros',
                 weight_attr=None, bias_attr=None, data_format='NCDHW'):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            weight_attr=nn.initializer.KaimingNormal(fan_in=out_channels * kernel_size * kernel_size),
            bias_attr=bias_attr,
            data_format=data_format
        )


class ResNeXtBottleneck(nn.Layer):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()

        mid_planes = cardinality * int(planes / 32)
        self.conv1 = Conv3D(inplanes, mid_planes, kernel_size=1, bias_attr=False)
        self.bn1 = BatchNorm3D(mid_planes)
        self.conv2 = Conv3D(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias_attr=False)
        self.bn2 = BatchNorm3D(mid_planes)
        self.conv3 = Conv3D(
            mid_planes, planes * self.expansion, kernel_size=1, bias_attr=False)
        self.bn3 = BatchNorm3D(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Layer):
    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 cardinality=32):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = Conv3D(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias_attr=False)
        self.bn1 = BatchNorm3D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3D(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)

        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)

        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)

        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        self.avgpool = nn.AvgPool3D((2, 1, 1), stride=1, exclusive=False)

    def _downsample_basic_block(self, x, planes, stride):
        out = fluid.layers.pool3d(x, pool_size=1, pool_stride=stride, pool_type='avg')
        shape = out.shape
        zero_pads = fluid.layers.zeros([shape[0], planes - shape[1], shape[2], shape[3], shape[4]],
                                       dtype='float32')
        out = fluid.layers.concat([out, zero_pads], axis=1)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(Conv3D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False), BatchNorm3D(planes * block.expansion))
        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resNext101():
    """Constructs a ResNext-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3])
    return model
