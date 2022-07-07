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
import paddle.nn.functional as F
from paddlevideo.utils import get_logger, gather_from_gpu, get_dist_info

from ...registry import RECOGNIZERS
from .base import BaseRecognizer

logger = get_logger("paddlevideo")


@RECOGNIZERS.register()
class RecognizerTransformer(BaseRecognizer):
    """Transformer's recognizer model framework."""
    def forward_net(self, imgs):
        # imgs.shape=[N,C,T,H,W], for transformer case
        if self.backbone is not None:
            feature = self.backbone(imgs)
        else:
            feature = imgs

        if self.head is not None:
            cls_score = self.head(feature)
        else:
            cls_score = None

        return cls_score

    def train_step(self, data_batch):
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
        labels_gathered = gather_from_gpu(labels) if world_size > 1 else labels
        cls_score_gathered = gather_from_gpu(
            cls_score) if world_size > 1 else cls_score

        # remove possible duplicate data
        real_data_size = kwargs.get("real_data_size")
        cur_data_size = len(labels_gathered)
        if real_data_size < cur_data_size:
            labels_gathered = labels_gathered[0:real_data_size]
            cls_score_gathered = cls_score_gathered[0:real_data_size]

        loss_metrics = self.head.loss(cls_score_gathered,
                                      labels_gathered,
                                      valid_mode=True,
                                      all_reduce=False)
        return loss_metrics

    def test_step(self, data_batch):
        """Define how the model is going to infer, from input to output."""
        imgs = data_batch[0]
        num_views = imgs.shape[2] // self.runtime_cfg.test.num_seg
        cls_score = []
        for i in range(num_views):
            view = imgs[:, :, i * self.runtime_cfg.test.num_seg:(i + 1) *
                        self.runtime_cfg.test.num_seg]
            cls_score.append(self.forward_net(view))
        cls_score = self._average_view(cls_score,
                                       self.runtime_cfg.test.avg_type)
        return cls_score

    def infer_step(self, data_batch):
        """Define how the model is going to infer, from input to output."""
        imgs = data_batch[0]
        num_views = imgs.shape[2] // self.runtime_cfg.test.num_seg
        cls_score = []
        for i in range(num_views):
            view = imgs[:, :, i * self.runtime_cfg.test.num_seg:(i + 1) *
                        self.runtime_cfg.test.num_seg]
            cls_score.append(self.forward_net(view))
        cls_score = self._average_view(cls_score,
                                       self.runtime_cfg.test.avg_type)
        return cls_score

    def _average_view(self, cls_score, avg_type='score'):
        """Combine the predicted results of different views

        Args:
            cls_score (list): results of multiple views
            avg_type (str, optional): Average calculation method. Defaults to 'score'.
        """
        assert avg_type in ['score', 'prob'], \
            f"Currently only the average of 'score' or 'prob' is supported, but got {avg_type}"
        if avg_type == 'score':
            return paddle.add_n(cls_score) / len(cls_score)
        elif avg_type == 'prob':
            return paddle.add_n(
                [F.softmax(score, axis=-1)
                 for score in cls_score]) / len(cls_score)
        else:
            raise NotImplementedError
