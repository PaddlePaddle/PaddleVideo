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

from ...registry import RECOGNIZERS
from .base import BaseRecognizer
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


@RECOGNIZERS.register()
class Recognizer3D(BaseRecognizer):
    """3D Recognizer model framework.
    """
    def forward(self, imgs, **kwargs):
        """Define how the model is going to run, from input to output.
        """
        feature = self.extract_feature(imgs)
        cls_score = self.head(feature)
        return cls_score

    def train_step(self, data_batch, reduce_sum=False):
        """Training step.
        """
        imgs = data_batch[0:2]
        labels = data_batch[2:]

        # call forward
        cls_score = self(imgs)
        loss_metrics = self.head.loss(cls_score, labels, reduce_sum=reduce_sum)
        return loss_metrics

    def val_step(self, data_batch, reduce_sum=True):
        """Validating setp.
        """
        return self.train_step(data_batch, reduce_sum=reduce_sum)

    def test_step(self, data_batch):
        """Test step.
        """
        imgs = data_batch[0:2]
        # call forward
        cls_score = self(imgs)

        return cls_score
