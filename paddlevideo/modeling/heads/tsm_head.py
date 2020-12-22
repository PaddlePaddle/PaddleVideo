# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F

from .tsn_head import TSNHead
from ..registry import HEADS

from ..weight_init import weight_init_


@HEADS.register()
class TSMHead(TSNHead):
    """ TSM Head

    Args:
        num_classes (int): The number of classes to be classified.
        in_channels (int): The number of channles in input feature.
        loss_cfg (dict): Config for building config. Default: dict(name='CrossEntropyLoss').
        drop_ratio(float): drop ratio. Default: 0.8.
        std(float): Std(Scale) value in normal initilizar. Default: 0.001.
        kwargs (dict, optional): Any keyword argument to initialize.
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 drop_ratio=0.8,
                 std=0.01,
                 **kwargs):

        super().__init__(num_classes,
                         in_channels,
                         drop_ratio=drop_ratio,
                         std=std,
                         **kwargs)

        self.stdv = 1.0 / math.sqrt(self.in_channels * 1.0)

    def init_weights(self):
        """Initiate the FC layer parameters"""

        weight_init_(self.fc,
                     'Uniform',
                     'fc_0.w_0',
                     'fc_0.b_0',
                     low=-self.stdv,
                     high=self.stdv)
        self.fc.bias.learning_rate = 2.0
        self.fc.bias.regularizer = paddle.regularizer.L2Decay(0.)
