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

from ...registry import SEGMENTERS
from .base import BaseSegmenter

import paddle
import paddle.nn.functional as F


@SEGMENTERS.register()
class MSTCN(BaseSegmenter):
    """MS-TCN model framework."""

    def forward_net(self, video_feature):
        """Define how the model is going to train, from input to output.
        """
        if self.backbone is not None:
            feature = self.backbone(video_feature)
        else:
            feature = video_feature

        if self.head is not None:
            cls_score = self.head(feature)
        else:
            cls_score = None

        return cls_score

    def train_step(self, data_batch):
        """Training step.
        """
        video_feat, video_gt = data_batch

        # call forward
        output = self.forward_net(video_feat)
        loss = 0.
        for i in range(len(output)):
            loss += self.head.loss(output[i], video_gt)

        predicted = paddle.argmax(output[-1], axis=1)
        predicted = paddle.squeeze(predicted)

        loss_metrics = dict()
        loss_metrics['loss'] = loss
        loss_metrics['F1@0.50'] = self.head.get_F1_score(predicted, video_gt)
        return loss_metrics

    def val_step(self, data_batch):
        """Validating setp.
        """
        video_feat, video_gt = data_batch

        # call forward
        output = self.forward_net(video_feat)
        loss = 0.
        for i in range(len(output)):
            loss += self.head.loss(output[i], video_gt)

        predicted = paddle.argmax(output[-1], axis=1)
        predicted = paddle.squeeze(predicted)

        outputs_dict = dict()
        outputs_dict['loss'] = loss
        outputs_dict['F1@0.50'] = self.head.get_F1_score(predicted, video_gt)
        return outputs_dict

    def test_step(self, data_batch):
        """Testing setp.
        """
        video_feat, _ = data_batch

        outputs_dict = dict()
        # call forward
        output = self.forward_net(video_feat)
        predicted = paddle.argmax(output[-1], axis=1)
        predicted = paddle.squeeze(predicted)
        outputs_dict['predict'] = predicted
        outputs_dict['output_np'] = F.sigmoid(output[-1])
        return outputs_dict

    def infer_step(self, data_batch):
        """Infering setp.
        """
        video_feat = data_batch[0]

        # call forward
        output = self.forward_net(video_feat)
        predicted = paddle.argmax(output[-1], axis=1)
        predicted = paddle.squeeze(predicted)
        output_np = F.sigmoid(output[-1])
        return predicted, output_np
