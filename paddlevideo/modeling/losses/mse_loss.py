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

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register()
class MSELoss(BaseWeightedLoss):
    """MSELoss Loss."""
    def __init__(self,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'sum',
                 **kwargs):
        super(MSELoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def _forward(self, input: paddle.Tensor,
                 label: paddle.Tensor) -> paddle.Tensor:
        loss = F.mse_loss(input, label, reduction=self.reduction)
        return loss