"""
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
"""
import math
import paddle
import paddle.nn.functional as F
from paddle.nn import AdaptiveAvgPool2D, Linear, Dropout
from .base import BaseHead
from .tsn_head import TSNHead
from ..registry import HEADS

from ..weight_init import weight_init_


@HEADS.register()
class TSMRecHead(TSNHead):
    """ TSM Rec Head

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
                 loss_cfg=dict(name='L1Loss'),
                 drop_ratio=0.8,
                 std=0.01,
                 data_format="NCHW",
                 **kwargs):

        super().__init__(num_classes,
                         in_channels,
                         loss_cfg,
                         drop_ratio=drop_ratio,
                         std=std,
                         data_format=data_format,
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

    def forward(self, x, num_seg):
        """Define how the head is going to run.

        Args:
            x (paddle.Tensor): The input data.
            num_segs (int): Number of segments.
        Returns:
            score: (paddle.Tensor) The classification scores for input samples.
        """
        # [N * num_segs, in_channels, 7, 7]
        x = self.avgpool2d(x)
        # [N * num_segs, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N * num_seg, in_channels, 1, 1]
        x = paddle.reshape(x, [-1, num_seg, x.shape[1]])
        # [N, num_seg, in_channels]
        x = paddle.mean(x, axis=1)
        # [N, 1, in_channels]
        x = paddle.reshape(x, shape=[-1, self.in_channels])
        # [N, in_channels]
        score = self.fc(x)
        # [N, num_class]
        #m = paddle.nn.Sigmoid()
        #score = m(score)
        return score

    def loss(self, scores, labels, valid_mode=False, **kwargs):
        """Calculate the loss accroding to the model output ```scores```,
           and the target ```labels```.

        Args:
            scores (paddle.Tensor): The output of the model.
            labels (paddle.Tensor): The target output of the model.

        Returns:
            losses (dict): A dict containing field 'loss'(mandatory).

        """
        if len(labels) == 1:  #commonly case
            output = []
            label = []
            labels = labels[0]
            losses = dict()
            loss = self.loss_func(scores, labels, **kwargs)

            score_list = paddle.tolist(scores)
            label_list = paddle.tolist(labels)
            score_list_len = len(score_list)
            for i in range(score_list_len):
                output.append(score_list[i][0])
                label.append(label_list[i][0])
            losses['loss'] = loss
            losses['output'] = output
            losses['label'] = label
            return losses
        elif len(labels) == 3:
            labels_a, labels_b, lam = labels
            labels_a = paddle.cast(labels_a, dtype='float32')
            labels_b = paddle.cast(labels_b, dtype='float32')
            lam = lam[0]  # get lam value
            losses = dict()

            if self.ls_eps != 0:
                loss_a = self.label_smooth_loss(scores, labels_a, **kwargs)
                loss_b = self.label_smooth_loss(scores, labels_b, **kwargs)
            else:
                loss_a = self.loss_func(scores, labels_a, **kwargs)
                loss_b = self.loss_func(scores, labels_a, **kwargs)
            loss = lam * loss_a + (1 - lam) * loss_b

            losses['loss'] = loss
            losses['output'] = output
            losses['label'] = label
            return losses
        else:
            raise NotImplementedError

    def label_smooth_loss(self, scores, labels, **kwargs):
        """label smooth loss"""
        labels = F.label_smooth(labels, epsilon=self.ls_eps)
        labels = paddle.squeeze(labels, axis=1)
        loss = self.loss_func(scores, labels, **kwargs)
        return loss
