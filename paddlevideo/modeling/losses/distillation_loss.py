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
import paddle.nn as nn

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register()
class DistillationCELoss(BaseWeightedLoss):
    """Distillation Entropy Loss."""
    def _forward(self, score, labels, **kwargs):
        """Forward function.
        Args:
            score (paddle.Tensor): The class score.
            labels (paddle.Tensor): The ground truth labels.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.
        Returns:
            loss (paddle.Tensor): The returned CrossEntropy loss.
        """
        if len(labels) == 1:
            label = labels[0]
            loss = F.cross_entropy(score, label, **kwargs)
        # Deal with VideoMix
        elif len(labels) == 3:
            label_a, label_b, lam = labels
            loss_a = F.cross_entropy(score, label_a, **kwargs)
            loss_b = F.cross_entropy(score, label_b, **kwargs)
            loss = lam * loss_a + (1 - lam) * loss_b
            loss = paddle.mean(loss)  #lam shape is bs
        return loss


@LOSSES.register()
class DistillationDMLLoss(BaseWeightedLoss):
    """
    DistillationDMLLoss
    """
    def __init__(self, act="softmax", eps=1e-12, **kargs):
        super().__init__(**kargs)
        if act is not None:
            assert act in ["softmax", "sigmoid"]
        if act == "softmax":
            self.act = nn.Softmax(axis=-1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None
        self.eps = eps

    def _kldiv(self, x, target):
        class_num = x.shape[-1]
        cost = target * paddle.log(
            (target + self.eps) / (x + self.eps)) * class_num
        return cost

    def _forward(self, x, target):
        if self.act is not None:
            x = self.act(x)
            target = self.act(target)
        loss = self._kldiv(x, target) + self._kldiv(target, x)
        loss = loss / 2
        loss = paddle.mean(loss)
        return loss
