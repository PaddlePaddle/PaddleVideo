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

from paddle.nn import Linear
import paddle

from ..registry import HEADS
from ..weight_init import trunc_normal_, weight_init_
from .base import BaseHead


@HEADS.register()
class TokenShiftHead(BaseHead):
    """TokenShift Transformer Head.

    Args:
        num_classes (int): The number of classes to be classified.
        in_channels (int): The number of channles in input feature.
        num_seg(int): The number of segments. Default: 8. 
        loss_cfg (dict): Config for building config. Default: dict(name='CrossEntropyLoss').
        ls_eps (float): Label smoothing epsilon. Default: 0.01.
        std (float): Std(Scale) Value in normal initilizar. Default: 0.02.
        kwargs (dict, optional): Any keyword argument to initialize.

    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_seg=8,
                 loss_cfg=dict(name='CrossEntropyLoss'),
                 ls_eps=0.01,
                 std=0.02,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cfg, ls_eps)
        self.num_seg = num_seg
        self.std = std
        self.fc = Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the FC layer parameters"""

        weight_init_(self.fc,
                     'TruncatedNormal',
                     'fc_0.w_0',
                     'fc_0.b_0',
                     mean=0.0,
                     std=self.std)
        # NOTE: Temporarily use trunc_normal_ instead of TruncatedNormal
        trunc_normal_(self.fc.weight, std=self.std)

    def forward(self, x):
        """Define how the head is going to run.
        Args:
            x (paddle.Tensor): The input data.
        Returns:
            score: (paddle.Tensor) The classification scores for input samples.
        """
        # XXX: check dropout location!
        # x.shape = [N, embed_dim]
        score = self.fc(x)
        # [N*T, num_class]
        _, _m = score.shape
        _t = self.num_seg
        score = score.reshape([-1, _t, _m])
        score = paddle.mean(score, 1)  # averaging predictions for every frame
        score = paddle.squeeze(score, axis=1)
        return score
