# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

import paddle.nn as nn

from .base import BaseHead
from ..registry import HEADS


@HEADS.register()
class STGCNPlusPlusHead(BaseHead):

    def __init__(self,
                 in_channels=256,
                 num_classes=10,
                 dropout=0.0,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)
        self.init_weights()

    def init_weights(self):
        nn.initializer.Normal(std=self.init_std, mean=0)(self.fc_cls.weight)
        nn.initializer.Constant(0)(self.fc_cls.bias)

    def forward(self, x):
        pool = nn.AdaptiveAvgPool2D(1)
        N, M, C, T, V = x.shape
        x = x.reshape([-1, C, T, V])

        x = pool(x)
        x = x.reshape([N, M, C])
        x = x.mean(axis=1)
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score
