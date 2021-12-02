# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..registry import BACKBONES


@BACKBONES.register()
class ExampleBackbone(nn.Layer):
    """Example Backbone"""
    def __init__(self):
        super(ExampleBackbone, self).__init__()
        ## 2.1 网络Backbobe
        self.layer1 = paddle.nn.Flatten(1, -1)
        self.layer2 = paddle.nn.Linear(400, 512)
        self.layer3 = paddle.nn.ReLU()
        self.layer4 = paddle.nn.Dropout(0.2)

    def forward(self, x):
        """ model forward"""
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        return y
