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

import paddle
import paddle.nn.functional as F

from .tsn_head import TSNHead
from ..registry import HEADS


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
                 num_class:int,
                 in_channels:int,
                 std:float,
                 **kwargs)->None:


        super().__init__(num_classes, in_channels, loss_cfg, loss_cfg, drop_ratio, std, **kwargs)
    """

        #NOTE: global pool performance
        self.avgpool2d = AdaptiveAvgPool2d((1,1))

        self.fc = Linear(
                    self.in_channels,
                    self.num_classes)

    def init_weight(self)->None:
        normal_init(self.fc, self.std)
   
    def forward(self, x:paddle.Tensor, seg_num:int)->paddle.Tensor:
     
        XXX

        Args:
            x (paddle.Tensor): The input data.
            num_segs (int): Number of segments.
        Returns:
            score: (paddle.Tensor) The classification scores for input samples.
        
        #XXX: [32, 2048, 7, 7] bs=4, seg_num=8
        # [N * seg_num, in_channels, 7, 7]
        x = self.avgpool2d(x)
        # [N * seg_num, in_channels, 1, 1]
        x = F.dropout(x, p=self.dropout_ratio)
        # [N * seg_num, in_channels, 1, 1]
        x = paddle.reshape(x, [-1, self.seg_num, x.shape[1]])
        # [N, seg_num, in_channels]
        x = paddle.mean(x, axis=1)
        # [N, 1, in_channels]
        x = paddle.reshape(x, shape=[-1, 2048])
        # [N, in_channels]
        score = self.fc(x)
        # [N, num_class]
        # XXX x = F.softmax(x)
        return score
    """

