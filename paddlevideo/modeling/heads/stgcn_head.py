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

from .base import BaseHead
from ..registry import HEADS


@HEADS.register()
class STGCNHead(BaseHead):
    """
    Head for ST-GCN and PP-AGCN model.
    Args:
        in_channels: int, input feature channels. Default: 256.
        num_classes: int, number classes. Default: 10.
    """
    def __init__(self, in_channels=256, num_classes=10, **kwargs):
        super().__init__(num_classes, in_channels, **kwargs)
        self.pool = nn.AdaptiveAvgPool2D(output_size=(1,
                                                      1))  # N,C,T,V --> N,C,1,1
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=in_channels, out_features=num_classes)

    def forward(self, x):
        """Define how the head is going to run.
        """
        x = self.pool(x)
        x = self.flatten(x)
        score = self.fc(x)
        return score
