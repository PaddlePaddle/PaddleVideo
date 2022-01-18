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
from paddlevideo.utils import load

import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import pandas as pd
import numpy as np


def unfold_1d(x, kernel_size=7, pad_value=0):
    """unfold_1d
    """
    B, C, T = x.shape
    padding = kernel_size // 2
    x = x.unsqueeze(-1)
    x = F.pad(x, (0, 0, padding, padding), value=pad_value)
    x = paddle.cast(x, 'float32')
    D = F.unfold(x, [kernel_size, 1])
    return paddle.reshape(D, [B, C, kernel_size, T])


def dual_barrier_weight(b, kernel_size=7, alpha=0.2):
    """dual_barrier_weight
    """
    K = kernel_size
    b = unfold_1d(b, kernel_size=K, pad_value=20)
    # b: (B, 1, K, T)
    HL = K // 2
    left = paddle.flip(
        paddle.cumsum(paddle.flip(b[:, :, :HL + 1, :], [2]), axis=2),
        [2])[:, :, :-1, :]
    right = paddle.cumsum(b[:, :, -HL - 1:, :], axis=2)[:, :, 1:, :]
    middle = paddle.zeros_like(b[:, :, 0:1, :])
    #middle = b[:, :, HL:-HL, :]
    weight = alpha * paddle.concat((left, middle, right), axis=2)
    return weight.neg().exp()


class LocalBarrierPooling(nn.Layer):
    """LocalBarrierPooling
    """

    def __init__(self, kernel_size=99, alpha=0.2):
        super(LocalBarrierPooling, self).__init__()
        self.kernel_size = kernel_size
        self.alpha = alpha

    def forward(self, x, barrier):
        """
        x: (B, C, T)
        barrier: (B, 1, T) (>=0)
        """
        xs = unfold_1d(x, self.kernel_size)
        w = dual_barrier_weight(barrier, self.kernel_size, self.alpha)
        return (xs * w).sum(axis=2) / ((w).sum(axis=2) + np.exp(-10))


@SEGMENTERS.register()
class BcnBgm(BaseSegmenter):
    """BCN model framework."""

    def forward_net(self, video_feature):
        """Define how the model is going to train, from input to output.
        """
        feature = self.backbone(video_feature)
        return feature

    def train_step(self, data_batch):
        """Training step.
        """
        feature, label = data_batch['feature_tensor'], data_batch['match_score']
        self.backbone.train()
        outputs = self.forward_net(feature)
        train_loss = self.loss(label, outputs)
        loss_metrics = {}
        loss_metrics['loss'] = train_loss

        return loss_metrics

    def val_step(self, data_batch):
        """Validating setp.
        """
        if isinstance(data_batch, dict):
            feature = data_batch['feature_tensor']
        elif isinstance(data_batch, list):
            feature = data_batch[0]
        self.backbone.eval()
        outputs = self.forward_net(feature)

        return outputs

    def test_step(self, data_batch):
        """Testing setp.
        """
        outputs = self.val_step(data_batch)
        self.head(outputs, data_batch['video_name'][0].split('.')[0])

        return outputs

    def infer_step(self, data_batch):
        """Infering setp.
        """
        outputs = self.val_step(data_batch)

        return outputs


@SEGMENTERS.register()
class BcnModel(BaseSegmenter):
    """BCN model framework.
    e.g.
        data_path = ./data/50salads/splits/train.split1.bundle
        bgm_result_path = ./output/BcnBgmResized/results
    """

    def __init__(self, data_path, bgm_result_path, bgm_pdparams, use_lbp,
                 num_post, **kwargs):
        super(BcnModel, self).__init__(**kwargs)
        # assert parameter
        assert '//' not in data_path, "don't use '//' in data_path, please use '/'"
        self.use_lbp = use_lbp
        self.bgm_result_path = bgm_result_path
        self.num_post = num_post

        self.iter = 0
        self.epoch = 0

        file_ptr = open(data_path, 'r')
        list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.epoch_iters = len(list_of_examples)

        dataset = data_path.split('/')[-3]
        freeze_epochs = 15
        pooling_length = 99
        if dataset == 'breakfast':
            freeze_epochs = 20
            pooling_length = 159
        elif dataset == 'gtea':
            freeze_epochs = 18
        self.freeze_epochs = freeze_epochs
        self.pooling_length = pooling_length
        self.dataset = dataset
        self.lbp = LocalBarrierPooling(pooling_length, alpha=2)
        self.backbone.bgm.set_state_dict(
            self.transformer_param_dict(load(bgm_pdparams)))

    def transformer_param_dict(self, param_dict):
        """transformer param_dict for bgm
        """
        new_param_dict = dict()
        for k in param_dict.keys():
            new_param_dict['.'.join(
                k.split('.')[2:])] = param_dict[k].cpu().detach().numpy()
        return new_param_dict

    def update_iter(self):
        """update_iter only use in train
        """
        if (self.epoch == 0) and (self.iter
                                  == 0) and (self.epoch <= self.freeze_epochs):
            self.freeze(self.backbone.bgm, True)

        self.iter += 1
        if self.iter >= self.epoch_iters:
            self.iter = 0
            self.epoch += 1
            if self.epoch > self.freeze_epochs:
                self.freeze(self.backbone.bgm, False)

    def freeze(self, sub_layer, flag):
        """freezing layer
        """
        for _, param in sub_layer.named_parameters():
            param.stop_gradient = flag

    def forward_net(self, batch_input, mask, gt_target=None):
        """Define how the model is going to train, from input to output.
        """
        outputs, BGM_output, output_weight = self.backbone(batch_input, mask)
        self.update_iter()
        return outputs, BGM_output, output_weight

    def train_step(self, data_batch):
        """Training step.
        """
        if isinstance(data_batch, dict):
            input_x, batch_target = data_batch['feature_tensor'], data_batch[
                'target_tensor']
            mask = data_batch['mask']
        elif isinstance(data_batch, list):
            input_x, batch_target = data_batch[0], data_batch[1]
            mask = data_batch[2]

        predictions, _, adjust_weight = self.forward_net(
            input_x, mask, batch_target)

        train_loss = self.loss(predictions, adjust_weight, batch_target, mask)
        loss_metrics = {}
        loss_metrics['loss'] = train_loss

        return loss_metrics

    def val_step(self, data_batch):
        """Validating setp.
        """
        if isinstance(data_batch, dict):
            input_x = data_batch['feature_tensor']
            mask = data_batch['mask']
            video_name = data_batch['video_name']
        elif isinstance(data_batch, list):
            input_x = data_batch[0]
            mask = data_batch[4]
            video_name = data_batch[5]
        predictions, _, _ = self.forward_net(input_x, mask)
        predictions = predictions[-1]
        if self.use_lbp and self.dataset != 'gtea':
            num_frames = np.shape(input_x)[2]
            if self.dataset in ['50salads', 'breakfast']:
                video_name = '/' + video_name[0].split('.')[0]
            barrier_file = self.bgm_result_path + video_name + ".csv"
            barrier = pd.read_csv(barrier_file)
            barrier = np.transpose(np.array(barrier))
            temporal_scale = np.shape(barrier)[1]
            barrier = paddle.to_tensor(barrier)

            if temporal_scale < num_frames:
                interpolation = paddle.round(
                    paddle.to_tensor([
                        float(num_frames) / temporal_scale * (i + 0.5)
                        for i in range(temporal_scale)
                    ]))
                interpolation = paddle.cast(interpolation, 'int64')
                resize_barrier = paddle.to_tensor([0.0] * num_frames)
                resize_barrier[interpolation] = barrier[0]
                resize_barrier = resize_barrier.unsqueeze(0).unsqueeze(0)
            else:
                resize_barrier = barrier
                resize_barrier = resize_barrier.unsqueeze(
                    0)  # size=[1,1,num_frames]
            if temporal_scale < num_frames:
                for i in range(self.num_post):
                    predictions = self.lbp(predictions, resize_barrier)
            else:
                predictions = F.interpolate(predictions, size=[temporal_scale], mode='linear', \
                    align_corners=False, data_format='NCW')
                for i in range(self.num_post):
                    predictions = self.lbp(predictions, resize_barrier)
                predictions = F.interpolate(predictions, size=[num_frames], mode='linear', \
                    align_corners=False, data_format='NCW')

        predicted = paddle.argmax(predictions, 1)
        predicted = predicted.squeeze()
        return predicted

    def test_step(self, data_batch):
        """Testing setp.
        """

        return self.val_step(data_batch)

    def infer_step(self, data_batch):
        """Infering setp.
        """
        # return self.val_step(data_batch)
        if isinstance(data_batch, list):
            input_x = data_batch[0]
            mask = data_batch[1]
        else:
            input_x = data_batch
            mask = paddle.ones([1, 1, input_x.shape[2]])

        predictions, _, _ = self.forward_net(input_x, mask)
        predicted = paddle.argmax(predictions, 1)
        predicted = predicted.squeeze()
        return predicted
