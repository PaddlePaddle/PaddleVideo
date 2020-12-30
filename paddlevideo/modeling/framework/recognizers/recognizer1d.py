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

@RECOGNIZERS.register()
class Recognizer1D(BaseRecognizer):
    """1D recognizer model framework."""

    def forward_train(self, imgs, labels, reduce_sum, use_mixup, **kwargs):
        """Define how the model is going to train, from input to output.
        """
        lstm_logit, lstm_output = self.head(imgs)

        loss = self.head.loss(lstm_logit,  labels, reduce_sum, **kwargs)

        hit_at_one, perr, gap = self.head.metric(lstm_output,  labels, reduce_sum, **kwargs)

        loss_metrics = dict()
        loss_metrics['loss'] = loss
        loss_metrics['hit_at_one'] = hit_at_one
        loss_metrics['perr'] = perr
        loss_metrics['gap'] = gap

        return loss_metrics

    def train_step(self, data_batch, **kwargs):
        """Training step.
        """
        imgs, labels = self.prepare_data(data_batch)

        # call forward
        loss_metrics = self.forward_train(imgs,
                                          labels,
                                          reduce_sum=False,
                                          use_mixup = 1,
                                          **kwargs)

        return loss_metrics

    def val_step(self, data_batch, **kwargs):
        """Validating setp.
        """
        imgs, labels = self.prepare_data(data_batch)

        # call forward
        loss_metrics = self.forward_train(imgs,
                                          labels,
                                          reduce_sum=True,
                                          use_mixup = 0,
                                          **kwargs)
        return loss_metrics

    def test_step(self, data_batch, **kwargs):
        imgs, labels = self.prepare_data(data_batch)

        metrics = self.forward_test(imgs, labels, reduce_sum=True, **kwargs)
        return  metrics

    def prepare_data(self, data):
        dtype="float32"
        x_data_rgb_tensor = paddle.cast(data[0],dtype)
        x_data_audio_tensor = paddle.cast(data[1],dtype)
        x_data_rgb_len_tensor = paddle.cast(data[2],dtype)
        x_data_audio_len_tensor = paddle.cast(data[3],dtype)
        x_data_rgb_mask_tensor = paddle.cast(data[4],dtype)
        x_data_audio_mask_tensor = paddle.cast(data[5],dtype)
        y_data_label_tensor = paddle.cast(data[6],dtype)
        imgs = [(x_data_rgb_tensor, x_data_rgb_len_tensor, x_data_rgb_mask_tensor),(x_data_audio_tensor, x_data_audio_len_tensor, x_data_audio_mask_tensor)]
        labels = y_data_label_tensor
        return imgs, labels
