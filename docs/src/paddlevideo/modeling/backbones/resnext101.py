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
import paddle


class ConvBNLayer(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 name=None,
                 data_format="NCDHW"):
        super(ConvBNLayer, self).__init__()
        self._conv = paddle.nn.Conv3D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingNormal(
                fan_in=num_filters * filter_size * filter_size), name=name+'_weights'),
            bias_attr=bias_attr,
            data_format=data_format)
        bn_name = "bn_" + name
        self._batch_norm = paddle.nn.BatchNorm3D(
            num_filters,
            momentum=0.9,
            epsilon=1e-05,
            weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(
                1.), name=bn_name + '_scale'),
            bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(
                0.), name=bn_name + '_offset'),
            data_format=data_format)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


def _downsample_basic_block(self, x, planes, stride):
    out = paddle.nn.functional.avg_pool3d(x, kernel_size=1, stride=stride)
    shape = out.shape
    zero_pads = paddle.zeros(shape=[shape[0], planes - shape[1], shape[2], shape[3], shape[4]],
                                   dtype='float32')
    out = paddle.concat(x=[out, zero_pads], axis=1)


class BottleneckBlock(paddle.nn.Layer):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None, name=None):
        super(BottleneckBlock, self).__init__()

        mid_planes = cardinality * int(planes / 32)
        self.conv0 = ConvBNLayer(
            inplanes, mid_planes, filter_size=1, bias_attr=False, name=name+'_branch2a')
        self.conv1 = ConvBNLayer(mid_planes, mid_planes, filter_size=3, stride=stride,
                                 padding=1, groups=cardinality, bias_attr=False, name=name+'_branch2b')
        self.conv2 = ConvBNLayer(mid_planes, planes * self.expansion,
                                 filter_size=1, bias_attr=False, name=name+'_branch2c')
        self.downsample = downsample
        self.stride = stride
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv0(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(paddle.nn.Layer):
    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 cardinality=32):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv = ConvBNLayer(
            3,
            64,
            filter_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias_attr=False,
            name="res_conv1"
        )
        self.relu = paddle.nn.ReLU()
        self.maxpool = paddle.nn.MaxPool3D(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality, stride=1, name='layer1')

        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2, name='layer2')

        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2, name='layer3')

        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2, name='layer4')
        self.avgpool = paddle.nn.AvgPool3D((2, 1, 1), stride=1, exclusive=False)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1,
                    name=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = ConvBNLayer(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias_attr=False,
                    name=name+'downsample'
                )
        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample, name=name+'_downsample'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                          cardinality, name=name+'_res_block'+str(i)))

        return paddle.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def ResNext101():
    """Constructs a ResNext-101 model.
    """
    model = ResNeXt(BottleneckBlock, [3, 4, 23, 3])
    return model
