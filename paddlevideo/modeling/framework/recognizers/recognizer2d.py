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
import paddle
from paddlevideo.utils import get_logger


logger = get_logger("paddlevideo")

@RECOGNIZERS.register()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Define how the model is going to train, from input to output.
        """
        # As the num_segs is an attribute of dataset phase, and didn't pass to build_head phase, should obtain it from imgs(paddle.Tensor) now, then call self.head method.
        
        batches = imgs.shape[0]
        imgs = paddle.reshape(imgs, [-1]+imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        feature = self.extract_feature(imgs)
        cls_score = self.head(feature, num_segs)
        #labels = labels.squeeze()
        #XXX: unsqueeze label to [label] ?
        loss_metrics = self.head.loss(cls_score, labels, **kwargs)
        return loss_metrics

    def forward_valid(self, imgs):
        """Define how the model is going to valid, from input to output."""
        #XXX add testing code.
        imgs = paddle.reshape(imgs, [-1]+imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        feature = self.extract_feature(imgs)
        cls_score = self.head(feature, num_segs)

        return cls_score
