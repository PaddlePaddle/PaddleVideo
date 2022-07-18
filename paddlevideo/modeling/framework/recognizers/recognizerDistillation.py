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

from abc import abstractmethod
import paddle
import paddle.nn as nn

from ...registry import RECOGNIZERS
from ... import builder
from paddlevideo.utils import get_logger, get_dist_info

logger = get_logger("paddlevideo")


@RECOGNIZERS.register()
class RecognizerDistillation(nn.Layer):
    """recognizer Distillation framework."""
    def __init__(self,
                 freeze_params_list=None,
                 models=None,
                 loss=None,
                 **kargs):
        """
        Args:
            freeze_params_list: list, set each model is trainable or not
            models: config of distillaciton model.
            loss: config of loss list
        """
        super().__init__()
        self.model_list = []
        self.model_name_list = []
        self.loss_cfgs = loss

        if freeze_params_list is None:
            freeze_params_list = [False] * len(models)
        assert len(freeze_params_list) == len(models)

        # build Teacher and Student model
        for idx, model_config in enumerate(models):
            assert len(model_config) == 1
            key = list(model_config.keys())[0]  #Teacher or Student
            model_config = model_config[key]
            model_name = model_config['backbone']['name']

            backbone, head = None, None
            if model_config.get('backbone'):
                backbone = builder.build_backbone(model_config['backbone'])
                if hasattr(backbone, 'init_weights'):
                    backbone.init_weights()
            if model_config.get('head'):
                head = builder.build_head(model_config['head'])
                if hasattr(head, 'init_weights'):
                    head.init_weights()

            model = nn.Sequential(backbone, head)
            logger.info('build distillation {} model done'.format(key))
            # for add all parameters in nn.Layer class
            self.model_list.append(self.add_sublayer(key, model))
            self.model_name_list.append({model_name: key})

            # set model trainable or not
            if freeze_params_list[idx]:
                for param in model.parameters():
                    param.trainable = False

        # build loss: support for loss list
        self.loss_func_list = []
        mode_keys = list(loss.keys())
        for mode in mode_keys:
            loss_cfgs = loss[mode]
            for loss_cfg in loss_cfgs:
                loss_func_dict = {}
                model_name_pairs = loss_cfg.pop('model_name_pairs')
                loss_func = builder.build_loss(loss_cfg)
                loss_func_dict['mode'] = mode
                loss_func_dict['loss_func'] = loss_func
                loss_func_dict['model_name_pairs'] = model_name_pairs
                self.loss_func_list.append(loss_func_dict)

    def forward(self, data_batch, mode='infer'):
        """
        1. Define how the model is going to run, from input to output.
        2. Console of train, valid, test or infer step
        3. Set mode='infer' is used for saving inference model, refer to tools/export_model.py
        """
        if mode == 'train':
            return self.train_step(data_batch)
        elif mode == 'valid':
            return self.val_step(data_batch)
        elif mode == 'test':
            return self.test_step(data_batch)
        elif mode == 'infer':
            return self.infer_step(data_batch)
        else:
            raise NotImplementedError

    def get_loss(self, output, labels, mode):
        """
        Args:
            output: dict, output name and its value
            labels: label of data
            mode: str, 'Train' or 'Val'
        """
        output['GroundTruth'] = labels
        loss_list = []

        for loss_func_dict in self.loss_func_list:
            if mode == loss_func_dict['mode']:
                model_name_pairs = loss_func_dict['model_name_pairs']
                loss_func = loss_func_dict['loss_func']
                loss_val = loss_func(output[model_name_pairs[0]],
                                     output[model_name_pairs[1]])
                loss_list.append(loss_val)

        total_loss = paddle.add_n(loss_list)
        return total_loss

    def get_acc(self, scores, labels, mode='Train'):
        def _get_acc(score, label, mode='Train'):
            top1 = paddle.metric.accuracy(input=score, label=label, k=1)
            top5 = paddle.metric.accuracy(input=score, label=label, k=5)
            _, world_size = get_dist_info()
            # Deal with multi cards validate
            if world_size > 1 and mode == 'Val':  #reduce sum when valid
                top1 = paddle.distributed.all_reduce(
                    top1, op=paddle.distributed.ReduceOp.SUM) / world_size
                top5 = paddle.distributed.all_reduce(
                    top5, op=paddle.distributed.ReduceOp.SUM) / world_size
            return top1, top5

        if len(labels) == 1:
            label = labels[0]
            return _get_acc(scores, label)
        # Deal with VideoMix
        elif len(labels) == 3:
            label_a, label_b, lam = labels
            top1a, top5a = _get_acc(scores, label_a, mode)
            top1b, top5b = _get_acc(scores, label_b, mode)
            top1 = lam * top1a + (1 - lam) * top1b
            top5 = lam * top5a + (1 - lam) * top5b
            return top1, top5

    def forward_model(self, imgs, model_name, model):
        if model_name in ['PPTSM_v2', 'ResNetTweaksTSM']:
            # [N,T,C,H,W] -> [N*T,C,H,W]
            imgs = paddle.reshape(imgs, [-1] + list(imgs.shape[2:]))

        return model(imgs)

    def train_step(self, data_batch):
        """Define how the model is going to train, from input to output.
        """
        out = {}
        loss_metrics = {}
        imgs = data_batch[0]
        labels = data_batch[1:]

        for idx, item in enumerate(self.model_name_list):
            model = self.model_list[idx]
            model_name = list(item.keys())[0]
            model_type = item[model_name]  # Teacher or Student
            out[model_type] = self.forward_model(imgs, model_name, model)

        # out_student, out_teacher
        loss = self.get_loss(out, labels, 'Train')
        loss_metrics['loss'] = loss
        # calculate acc with student output
        top1, top5 = self.get_acc(out['Student'], labels)
        loss_metrics['top1'] = top1
        loss_metrics['top5'] = top5

        return loss_metrics

    def val_step(self, data_batch):
        out = {}
        loss_metrics = {}
        imgs = data_batch[0]
        labels = data_batch[1:]

        for idx, item in enumerate(self.model_name_list):
            model = self.model_list[idx]
            model_name = list(item.keys())[0]
            model_type = item[model_name]  # Teacher or Student
            out[model_type] = self.forward_model(imgs, model_name, model)

        # Loss of student with gt:  out_student, label
        loss = self.get_loss(out, labels, 'Val')
        loss_metrics['loss'] = loss

        top1, top5 = self.get_acc(out['Student'], labels, 'Val')
        loss_metrics['top1'] = top1
        loss_metrics['top5'] = top5

        return loss_metrics

    def test_step(self, data_batch):
        """Define how the model is going to test, from input to output."""
        imgs = data_batch[0]

        # Use Student to test
        for idx, item in enumerate(self.model_name_list):
            model = self.model_list[idx]
            model_name = list(item.keys())[0]
            model_type = item[model_name]  # Teacher or Student
            if model_type == "Student":
                out = self.forward_model(imgs, model_name, model)

        return out

    def infer_step(self, data_batch):
        """Define how the model is going to test, from input to output."""
        imgs = data_batch[0]

        # Use Student to infer
        for idx, item in enumerate(self.model_name_list):
            model = self.model_list[idx]
            model_name = list(item.keys())[0]
            model_type = item[model_name]  # Teacher or Student
            if model_type == "Student":
                out = self.forward_model(imgs, model_name, model)

        return out
