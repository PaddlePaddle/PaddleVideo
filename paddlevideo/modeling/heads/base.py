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
from abc import abstractmethod

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
        ls_eps (float): label smoothing epsilon. Default: 0. .

    """
    def __init__(
        self,
        num_classes,
        in_channels,
        loss_cfg=dict(
            name="CrossEntropyLoss"
        ),  #TODO(shipping): only pass a name or standard build cfg format.
        #multi_class=False, NOTE(shipping): not supported now.
        ls_eps=0.):

        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_func = build_loss(loss_cfg)
        #self.multi_class = multi_class NOTE(shipping): not supported now
        self.ls_eps = ls_eps

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters.
        """
        raise NotImplemented

    @abstractmethod
    def forward(self, x):
        """Define how the head is going to run.
        """
        raise NotImplemented

    def loss(self, scores, labels, reduce_sum=False, **kwargs):
        """Calculate the loss accroding to the model output ```scores```,
           and the target ```labels```.

        Args:
            scores (paddle.Tensor): The output of the model.
            labels (paddle.Tensor): The target output of the model.

        Returns:
            losses (dict): A dict containing field 'loss'(mandatory) and 'top1_acc', 'top5_acc'(optional).

        """
        if len(labels) == 1:
            labels = labels[0]
        elif len(labels) == 3:
            labels_a, labels_b, lam = labels
            return self.mixup_loss(scores, labels_a, labels_b, lam)
        else:
            raise NotImplemented

        if self.ls_eps != 0.:
            labels = F.one_hot(labels, self.num_classes)
            labels = F.label_smooth(labels, epsilon=self.ls_eps)
            # reshape [bs, 1, num_classes] to [bs, num_classes]
            #NOTE: maybe squeeze is helpful for understanding.
            labels = paddle.reshape(labels, shape=[-1, self.num_classes])
        #labels.stop_gradient = True  #XXX(shipping): check necessary
        losses = dict()
        #NOTE(shipping): F.crossentropy include logsoftmax and nllloss !
        #NOTE(shipping): check the performance of F.crossentropy
        loss = self.loss_func(scores, labels, **kwargs)
        avg_loss = paddle.mean(loss)
        top1 = paddle.metric.accuracy(input=scores, label=labels, k=1)
        top5 = paddle.metric.accuracy(input=scores, label=labels, k=5)

        _, world_size = get_dist_info()

        #NOTE(shipping): deal with multi cards validate
        if world_size > 1 and reduce_sum:
            top1 = paddle.distributed.all_reduce(
                top1, op=paddle.distributed.ReduceOp.SUM) / world_size
            top5 = paddle.distributed.all_reduce(
                top5, op=paddle.distributed.ReduceOp.SUM) / world_size

        losses['top1'] = top1
        losses['top5'] = top5
        losses['loss'] = avg_loss

        return losses

    def mixup_loss(self, scores, labels_a, labels_b, lam):
        if self.ls_eps != 0:
            labels_a = F.one_hot(labels_a, self.num_classes)
            labels_a = F.label_smooth(labels_a, epsilon=self.ls_eps)
            labels_b = F.one_hot(labels_b, self.num_classes)
            labels_b = F.label_smooth(labels_b, epsilon=self.ls_eps)
            # reshape [bs, 1, num_classes] to [bs, num_classes]
            labels_a = paddle.reshape(labels_a, shape=[-1, self.num_classes])
            labels_b = paddle.reshape(labels_b, shape=[-1, self.num_classes])

        losses = dict()
        loss_a = self.loss_func(scores, labels_a, soft_label=True)
        loss_b = self.loss_func(scores, labels_b, soft_label=True)
        avg_loss_a = paddle.mean(
            loss_a)  #FIXME: call mean here or in last step?
        avg_loss_b = paddle.mean(loss_b)
        avg_loss = lam * avg_loss_a + (1 - lam) * avg_loss_b
        losses['loss'] = avg_loss

        return losses
