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
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Args:
        class_weight (list, optional): per-class rescaling weight used to
            counter class imbalance, e.g. inverse class frequency. Length
            must match num_classes. Default: None (no reweighting).
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
    """
    def __init__(self, class_weight=None, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = paddle.to_tensor(class_weight,
                                                  dtype='float32')

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
        if self.class_weight is not None and 'weight' not in kwargs:
            kwargs['weight'] = self.class_weight
        loss = F.cross_entropy(score, labels, **kwargs)
        return loss
