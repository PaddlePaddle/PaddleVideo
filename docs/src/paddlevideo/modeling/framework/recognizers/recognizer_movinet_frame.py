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

import paddle

from paddlevideo.utils import get_logger
from .base import BaseRecognizer
from ...registry import RECOGNIZERS

logger = get_logger("paddlevideo")


@RECOGNIZERS.register()
class MoViNetRecognizerFrame(BaseRecognizer):

    def forward_net(self, imgs):
        """Define how the model is going to run, from input to output.
        """
        self.backbone.clean_activation_buffers()
        outputs = self.backbone(imgs)
        cls_score = self.head(outputs)
        return cls_score

    def train_step(self, data_batch):
        """Training step.
        """
        imgs = data_batch[0]
        labels = data_batch[1]  #.astype("int64")
        data = paddle.transpose(imgs, perm=[0, 2, 1, 3, 4])
        # call forward
        cls_score = self.forward_net(data)
        loss_metrics = self.head.loss_func(cls_score, labels)
        top1 = paddle.metric.accuracy(input=cls_score, label=labels, k=1)
        top5 = paddle.metric.accuracy(input=cls_score, label=labels, k=5)
        output = {'loss': loss_metrics, 'top1': top1, 'top5': top5}
        return output

    def val_step(self, data_batch):
        """Validating setp.
        """
        imgs = data_batch[0]
        labels = data_batch[1]  #.astype("int64")
        data = paddle.transpose(imgs, perm=[0, 2, 1, 3, 4])
        # call forward
        cls_score = self.forward_net(data)
        loss_metrics = self.head.loss_func(cls_score, labels)
        top1 = paddle.metric.accuracy(input=cls_score, label=labels, k=1)
        top5 = paddle.metric.accuracy(input=cls_score, label=labels, k=5)
        output = {'loss': loss_metrics, 'top1': top1, 'top5': top5}
        return output

    def test_step(self, data_batch):
        """Test step.
        """
        imgs = data_batch[0]
        data = paddle.transpose(imgs, perm=[0, 2, 1, 3, 4])
        # call forward
        cls_score = self.forward_net(data)
        return cls_score

    def infer_step(self, data_batch):
        """Infer step.
        """
        imgs = data_batch[0]
        # call forward
        data = paddle.transpose(imgs, perm=[0, 2, 1, 3, 4])
        cls_score = self.forward_net(data)

        return cls_score
