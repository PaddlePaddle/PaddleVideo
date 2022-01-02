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

from paddle import ParamAttr

from .base import BaseHead
from ..registry import HEADS
from ..weight_init import weight_init_


@HEADS.register()
class MSTCNHead(BaseHead):

    def __init__(self, num_classes, in_channels):
        super().__init__(num_classes, in_channels)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def forward(self, x):
        """MS-TCN no head
        """
        return x

    def loss(self, output, video_gt):
        """calculate loss
        """
        output_transpose = paddle.transpose(output, [2, 0, 1])
        ce_x = paddle.reshape(output_transpose,
                              (output_transpose.shape[0] *
                               output_transpose.shape[1], self.num_classes))
        ce_y = video_gt[0, :]
        ce_loss = self.ce(ce_x, ce_y)
        loss = ce_loss

        mse = self.mse(F.log_softmax(output[:, :, 1:], axis=1),
                       F.log_softmax(output.detach()[:, :, :-1], axis=1))
        mse = paddle.clip(mse, min=0, max=16)
        mse_loss = 0.15 * paddle.mean(mse)
        loss += mse_loss

        return loss
