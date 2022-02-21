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

import paddle
import paddle.nn.functional as F
from .utils import ASRFPostProcessing


@SEGMENTERS.register()
class ASRF(BaseSegmenter):
    """ASRF model framework."""

    def __init__(self,
                 postprocessing_method,
                 boundary_threshold,
                 backbone=None,
                 head=None,
                 loss=None):

        super().__init__(backbone=backbone, head=head, loss=loss)
        self.postprocessing_method = postprocessing_method
        self.boundary_threshold = boundary_threshold

    def forward_net(self, video_feature):
        """Define how the model is going to train, from input to output.
        """
        if self.backbone is not None:
            feature = self.backbone(video_feature)
        else:
            feature = video_feature

        if self.head is not None:
            network_outputs = self.head(feature)
        else:
            network_outputs = None

        return network_outputs

    def train_step(self, data_batch):
        """Training step.
        """
        feature, label, boundary = data_batch
        # call forward
        outputs_cls, outputs_boundary = self.forward_net(feature)

        # transfer data
        outputs_cls_np = outputs_cls[-1].numpy()
        outputs_boundary_np = outputs_boundary[-1].numpy()

        # caculate loss
        if self.loss is not None:
            output_loss = self.loss(feature, outputs_cls, label,
                                    outputs_boundary, boundary)
        else:
            output_loss = None

        # predict post process
        predicted = ASRFPostProcessing(outputs_cls_np, outputs_boundary_np,
                                       self.postprocessing_method)
        predicted = paddle.squeeze(predicted)

        loss_metrics = dict()
        loss_metrics['loss'] = output_loss
        loss_metrics['F1@0.50'] = self.head.get_F1_score(predicted, label)

        return loss_metrics

    def val_step(self, data_batch):
        """Validating setp.
        """
        feature, label, boundary = data_batch

        # call forward
        outputs_cls, outputs_boundary = self.forward_net(feature)

        # transfer data
        outputs_cls_np = outputs_cls[-1].numpy()
        outputs_boundary_np = outputs_boundary[-1].numpy()

        ## caculate loss
        if self.loss is not None:
            output_loss = self.loss(feature, outputs_cls, label,
                                    outputs_boundary, boundary)
        else:
            output_loss = None

        # predict post process
        predicted = ASRFPostProcessing(outputs_cls_np, outputs_boundary_np,
                                       self.postprocessing_method)
        predicted = paddle.squeeze(predicted)

        outputs_dict = dict()
        outputs_dict['loss'] = output_loss
        outputs_dict['F1@0.50'] = self.head.get_F1_score(predicted, label)
        return outputs_dict

    def test_step(self, data_batch):
        """Testing setp.
        """
        feature, _, _ = data_batch

        outputs_dict = dict()
        # call forward
        outputs_cls, outputs_boundary = self.forward_net(feature)
        # transfer data
        outputs_cls_np = outputs_cls[-1].numpy()
        outputs_boundary_np = outputs_boundary[-1].numpy()

        # predict post process
        predicted = ASRFPostProcessing(outputs_cls_np, outputs_boundary_np,
                                       self.postprocessing_method)
        outputs_dict['predict'] = paddle.to_tensor(predicted[0, :])
        outputs_dict['output_np'] = F.sigmoid(outputs_cls[-1])
        return outputs_dict

    def infer_step(self, data_batch):
        """Infering setp.
        """
        feature = data_batch[0]

        # call forward
        outputs_cls, outputs_boundary = self.forward_net(feature)
        # transfer data
        outputs_cls_np = outputs_cls[-1]
        outputs_boundary_np = outputs_boundary[-1]

        outputs = [
            outputs_cls_np, outputs_boundary_np,
            F.sigmoid(outputs_cls[-1])
        ]
        return outputs
