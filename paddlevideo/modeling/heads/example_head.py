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
from ..registry import HEADS
from .base import BaseHead


@HEADS.register()
class ExampleHead(BaseHead):
    """ Example Head """
    def __init__(self, num_classes=10, in_channels=512):
        super().__init__(num_classes, in_channels)
        self.head = paddle.nn.Linear(in_channels, num_classes)

    def forward(self, x):
        """model forward """
        y = self.head(x)
        return y
