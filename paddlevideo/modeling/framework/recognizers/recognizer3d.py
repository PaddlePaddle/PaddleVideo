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
    """Recognizer3D.

    All recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_valid``, supporting to forward when validating.

    Args:
        backbone (dict): Backbone modules to extract feature.
        head (dict): Classification head to process feature.
        #XXX cfg keep or not????
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.

    """
    def forward_train(self, imgs, labels, reduce_sum, **kwargs):
        cls_score = self(imgs)
        loss_metrics = self.head.loss(cls_score, labels, reduce_sum, **kwargs)
        return loss_metrics

    def forward_test(self, imgs, **kwargs):
        cls_score = self(imgs)
        return cls_score

    def forward(self, imgs, **kwargs):
        """Define how the model is going to run, from input to output.
        """
        feature = self.extract_feature(imgs)
        cls_score = self.head(feature)
        return cls_score

    def train_step(self, data_batch, **kwargs):
        """Training step.
        """
        imgs = [data_batch[0], data_batch[1]]
        labels = data_batch[2]

        # call forward
        loss_metrics = self.forward_train(imgs,
                                          labels,
                                          reduce_sum=False,
                                          **kwargs)

        return loss_metrics

    def val_step(self, data_batch, **kwargs):
        """Validating setp.
        """
        imgs = [data_batch[0], data_batch[1]]
        labels = data_batch[2]

        # call forward
        loss_metrics = self.forward_train(imgs,
                                          labels,
                                          reduce_sum=True,
                                          **kwargs)
        return loss_metrics

    def test_step(self, data_batch, **kwargs):
        """Test step.
        """
        imgs = [data_batch[0], data_batch[1]]

        # call forward
        loss_metrics = self.forward_test(imgs, **kwargs)

        return loss_metrics
