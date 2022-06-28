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
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import copy
import random
import math

from paddle import ParamAttr
from ..registry import BACKBONES
from ..weight_init import weight_init_


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = len(tensor.shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed \
        for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    else:
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def calculate_gain(nonlinearity=None, a=None):
    if nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if a != None:
            return math.sqrt(2.0 / (1 + a**2))
        else:
            return math.sqrt(2.0 / (1 + 0.01**2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        return 1


def KaimingUniform_like_torch(weight_npy,
                              mode='fan_in',
                              nonlinearity='leaky_relu'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(weight_npy)
    if mode == 'fan_in':
        fan_mode = fan_in
    else:
        fan_mode = fan_out
    a = math.sqrt(5.0)
    gain = calculate_gain(nonlinearity=nonlinearity, a=a)
    std = gain / math.sqrt(fan_mode)
    bound = math.sqrt(3.0) * std
    return np.random.uniform(-bound, bound, weight_npy.shape)


def init_bias(weight_npy, bias_npy):
    # attention this weight is not bias
    fan_in, fan_out = _calculate_fan_in_and_fan_out(weight_npy)
    bound = 1.0 / math.sqrt(fan_in)
    return np.random.uniform(-bound, bound, bias_npy.shape)


class SingleStageModel(nn.Layer):

    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_in = nn.Conv1D(dim, num_f_maps, 1)
        self.layers = nn.LayerList([
            copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps))
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1D(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Layer):

    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1D(in_channels,
                                      out_channels,
                                      3,
                                      padding=dilation,
                                      dilation=dilation)
        self.conv_in = nn.Conv1D(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)
        return (x + out)


@BACKBONES.register()
class MSTCN(nn.Layer):

    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super().__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.LayerList([
            copy.deepcopy(
                SingleStageModel(num_layers, num_f_maps, num_classes,
                                 num_classes)) for s in range(num_stages - 1)
        ])

    def forward(self, x):
        """ MSTCN forward
        """
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, axis=1))
            outputs = paddle.concat((outputs, out.unsqueeze(0)), axis=0)
        return outputs

    def init_weights(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv1D):
                layer.weight.set_value(
                    KaimingUniform_like_torch(layer.weight).astype('float32'))
                if layer.bias is not None:
                    layer.bias.set_value(
                        init_bias(layer.weight, layer.bias).astype('float32'))
