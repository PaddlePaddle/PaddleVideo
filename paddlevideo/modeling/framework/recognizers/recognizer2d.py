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
from paddlevideo.utils import get_logger, gather_from_gpu, get_dist_info

logger = get_logger("paddlevideo")


@RECOGNIZERS.register()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""
    def forward_net(self, imgs):
        # NOTE: As the num_segs is an attribute of dataset phase, and didn't pass to build_head phase, should obtain it from imgs(paddle.Tensor) now, then call self.head method.
        num_segs = imgs.shape[
            1]  # imgs.shape=[N,T,C,H,W], for most commonly case
        imgs = paddle.reshape_(imgs, [-1] + list(imgs.shape[2:]))

        if self.backbone is not None:
            feature = self.backbone(imgs)
        else:
            feature = imgs

        if self.head is not None:
            cls_score = self.head(feature, num_segs)
        else:
            cls_score = None

        return cls_score

    def train_step(self, data_batch, **kwargs):
        """Define how the model is going to train, from input to output.
        """
        imgs = data_batch[0]
        labels = data_batch[1:]
        cls_score = self.forward_net(imgs)
        loss_metrics = self.head.loss(cls_score, labels)
        return loss_metrics

    def val_step(self, data_batch, **kwargs):
        imgs = data_batch[0]
        labels = data_batch[1:]
        cls_score = self.forward_net(imgs)

        # gather data from all devices
        _, world_size = get_dist_info()
        labels_gathered = [gather_from_gpu(x)
                           for x in labels] if world_size > 1 else labels
        cls_score_gathered = gather_from_gpu(
            cls_score) if world_size > 1 else cls_score

        # remove possible duplicate data
        real_data_size = kwargs.get("real_data_size")
        cur_data_size = len(labels_gathered)
        if real_data_size < cur_data_size:
            labels_gathered = [x[0:real_data_size] for x in labels_gathered]
            cls_score_gathered = cls_score_gathered[0:real_data_size]

        loss_metrics = self.head.loss(cls_score_gathered,
                                      labels_gathered,
                                      valid_mode=True,
                                      all_reduce=False)
        return loss_metrics

    def test_step(self, data_batch, **kwargs):
        """Define how the model is going to test, from input to output."""
        # NOTE: (shipping) when testing, the net won't call head.loss, we deal with the test processing in /paddlevideo/metrics
        imgs = data_batch[0]
        cls_score = self.forward_net(imgs)
        return cls_score

    def infer_step(self, data_batch, **kwargs):
        """Define how the model is going to test, from input to output."""
        imgs = data_batch[0]
        cls_score = self.forward_net(imgs)
        return cls_score
