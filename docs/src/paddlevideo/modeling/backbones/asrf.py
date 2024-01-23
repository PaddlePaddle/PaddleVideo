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

# https://github.com/yabufarha/ms-tcn/blob/master/model.py
# https://github.com/yiskw713/asrf/libs/models/tcn.py

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
from .ms_tcn import DilatedResidualLayer
from ..framework.segmenters.utils import init_bias, KaimingUniform_like_torch


@BACKBONES.register()
class ASRF(nn.Layer):

    def __init__(self, in_channel, num_features, num_classes, num_stages,
                 num_layers):
        super().__init__()
        self.in_channel = in_channel
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.num_layers = num_layers

        # define layers
        self.conv_in = nn.Conv1D(self.in_channel, self.num_features, 1)

        shared_layers = [
            DilatedResidualLayer(2**i, self.num_features, self.num_features)
            for i in range(self.num_layers)
        ]
        self.shared_layers = nn.LayerList(shared_layers)

        self.init_weights()

    def init_weights(self):
        """
        initialize model layers' weight
        """
        # init weight
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv1D):
                layer.weight.set_value(
                    KaimingUniform_like_torch(layer.weight).astype('float32'))
                if layer.bias is not None:
                    layer.bias.set_value(
                        init_bias(layer.weight, layer.bias).astype('float32'))

    def forward(self, x):
        """ ASRF forward
        """
        out = self.conv_in(x)
        for layer in self.shared_layers:
            out = layer(out)
        return out
