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

import math
import paddle
import paddle.nn as nn

from .base import BaseHead
from ..registry import HEADS
from ..weight_init import weight_init_


@HEADS.register()
class AGCN2sHead(BaseHead):
    """
    Head for AGCN2s model.
    Args:
        in_channels: int, input feature channels. Default: 64.
        num_classes: int, output the number of classes.
        M: int, number of people.
        drop_out: float, dropout ratio of layer. Default: 0.
    """
    def __init__(self, in_channels=64, num_classes=10, M=2, **kwargs):
        super().__init__(num_classes, in_channels, **kwargs)
        self.in_channels = in_channels
        self.M = M
        weight_attr = paddle.ParamAttr(
            name="linear_weight",
            initializer=paddle.nn.initializer.Normal(mean=0.0,
                                                     std=math.sqrt(
                                                         2. / num_classes)))

        self.fc = nn.Linear(self.in_channels * 4,
                            self.num_classes,
                            weight_attr=weight_attr)

    def forward(self, x):
        """Define how the head is going to run.
        """
        assert x.shape[
            0] % self.M == 0, f'The first dimension of the output must be an integer multiple of the number of people M, but recieved shape[0]={x.shape[0]}, M={self.M}'
        # N*M,C,T,V
        N = x.shape[0] // self.M
        c_new = x.shape[1]
        x = x.reshape([N, self.M, c_new, -1])
        x = x.mean(3).mean(1)

        return self.fc(x)
