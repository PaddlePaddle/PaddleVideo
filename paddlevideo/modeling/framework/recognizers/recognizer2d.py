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
import sys
import numpy as np


logger = get_logger("paddlevideo")

@RECOGNIZERS.register()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def forward_train(self, imgs, labels, reduce_sum, use_mixup, **kwargs):
        """Define how the model is going to train, from input to output.
        """
        #NOTE: As the num_segs is an attribute of dataset phase, and didn't pass to build_head phase, should obtain it from imgs(paddle.Tensor) now, then call self.head method.

        #labels = labels.squeeze()
        #XXX: unsqueeze label to [label] ?
        # mixup step 1/2 
        if use_mixup == 1:
            mix_images = imgs.numpy()  #(8, 8, 3, 224, 224) 
            mix_target = labels.numpy()  
            mix_bs = mix_images.shape[0]
            mix_alpha=0.2 
            mix_lam = np.random.beta(mix_alpha,mix_alpha)
            mix_index = np.random.permutation(mix_bs)  #rand 0~N-1, e.g. [3 6 0 2 4 7 5 1] for N=bs=8
            mix_inputs = mix_lam*mix_images + (1-mix_lam)*mix_images[mix_index]
            mix_targets_a, mix_targets_b = mix_target, mix_target[mix_index]
            imgs = paddle.to_tensor(mix_inputs)
            labels_a = paddle.to_tensor(mix_targets_a.reshape([-1, 1]))
            labels_b = paddle.to_tensor(mix_targets_b.reshape([-1, 1]))
            labels_a.stop_gradient = True
            labels_b.stop_gradient = True
        
        cls_score = self(imgs)

        # mixup step 2/2 
        if use_mixup == 1:
            loss_metrics_a = self.head.loss(cls_score, labels_a, reduce_sum, **kwargs)
            loss_metrics_b = self.head.loss(cls_score, labels_b, reduce_sum, **kwargs)
            avg_loss = mix_lam * loss_metrics_a['loss'] + (1 - mix_lam) * loss_metrics_b['loss']
            loss_metrics = loss_metrics_a
            loss_metrics['loss'] = avg_loss
        else:
            loss_metrics = self.head.loss(cls_score, labels, reduce_sum, **kwargs)
        return loss_metrics

    def forward_test(self, imgs, labels, reduce_sum, **kwargs):
        """Define how the model is going to test, from input to output."""
        #XXX
        num_segs = imgs.shape[1]
        cls_score = self(imgs)

        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score, num_segs)
        metrics = self.head.loss(cls_score, labels, reduce_sum, return_loss=False **kwargs)

        return metrics
