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


import numpy as np
from abc import  abstractmethod

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..builder import build_loss
from paddlevideo.utils import get_logger, get_dist_info 

logger = get_logger("paddlevideo")
class BaseHead(nn.Layer):
    """Base class for head part.

    All head should subclass it.
    All subclass should overwrite:

    - Methods: ```init_weights```, initializing weights.
    - Methods: ```forward```, forward function.

    Args:
        num_classes (int): The number of classes to be classified.
        in_channels (int): The number of channels in input feature.
        loss_cfg (dict): Config for building loss. Default: dict(type='CrossEntropyLoss').

    """ 

    def __init__(self,
		 num_classes,
		 in_channels,
                 loss_cfg=dict(name="CrossEntropyLoss")
	         ):

        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_func = build_loss(loss_cfg)

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters.
        """
        pass

    @abstractmethod
    def forward(self, x):
        """Define how the head is going to run.
        """
        pass

    def loss(self, scores, labels, reduce_sum=False, return_loss=False, **kwargs):
        """Calculate the loss accroding to the model output ```scores```, 
           and the target ```labels```.

        Args:
            scores (paddle.Tensor): The output of the model.
            labels (paddle.Tensor): The target output of the model.

        Returns:
            losses (dict): A dict containing field 'loss'(mandatory) and 'top1_acc', 'top5_acc'(optional).

        """
        labels.stop_gradient=True #XXX: check necessary
        losses = dict()
        if return_loss:
            #XXX: F.crossentropy include logsoftmax and nllloss 
            loss = self.loss_func(scores, labels, **kwargs)
            avg_loss = paddle.mean(loss)
        top1 = paddle.metric.accuracy(input=scores, label=labels, k=1)
        top5 = paddle.metric.accuracy(input=scores, label=labels, k=5)

        _, world_size = get_dist_info()

        #deal with multi cards validate
        if world_size > 1:
            top1 = paddle.distributed.all_reduce(top1, op=paddle.distributed.ReduceOp.SUM)/ world_size
            top5 = paddle.distributed.all_reduce(top5, op=paddle.distributed.ReduceOp.SUM)/ world_size

        losses['top1'] = top1
        losses['top5'] = top5
        if return_loss:
            if type(loss) is dict:
                losses.update(avg_loss)
            else:
                losses['loss'] = avg_loss

        return losses
