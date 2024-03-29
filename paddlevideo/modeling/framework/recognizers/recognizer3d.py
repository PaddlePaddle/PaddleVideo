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

    def forward_net(self, imgs):
        """Define how the model is going to run, from input to output.
        """
        feature = self.backbone(imgs)
        cls_score = self.head(feature)
        return cls_score

    def train_step(self, data_batch):
        """Training step.
        """
        if self.backbone.__class__.__name__ == 'ResNet3dSlowOnly':
            imgs = data_batch[0]
            labels = data_batch[1:]
            if imgs.dim() == 6:
                imgs = imgs.reshape([-1] + imgs.shape[2:])
        else:
            imgs = data_batch[0:2]
            labels = data_batch[2:]

        # call forward
        cls_score = self.forward_net(imgs)
        loss_metrics = self.head.loss(cls_score, labels)
        return loss_metrics

    def val_step(self, data_batch):
        """Validating setp.
        """
        if self.backbone.__class__.__name__ == 'ResNet3dSlowOnly':
            imgs = data_batch[0]
            labels = data_batch[1:]
            if imgs.dim() == 6:
                imgs = imgs.reshape([-1] + imgs.shape[2:])
        else:
            imgs = data_batch[0:2]
            labels = data_batch[2:]

        # call forward
        cls_score = self.forward_net(imgs)
        loss_metrics = self.head.loss(cls_score, labels, valid_mode=True)
        return loss_metrics

    def test_step(self, data_batch):
        """Test step.
        """
        if self.backbone.__class__.__name__ == 'ResNet3dSlowOnly':
            imgs = data_batch[0]
            if imgs.dim() == 6:
                imgs = imgs.reshape([-1] + imgs.shape[2:])
        else:
            imgs = data_batch[0:2]
        # call forward
        cls_score = self.forward_net(imgs)

        return cls_score

    def infer_step(self, data_batch):
        """Infer step.
        """
        if self.backbone.__class__.__name__ == 'ResNet3dSlowOnly':
            imgs = data_batch[0]
            # call forward
            imgs = imgs.reshape([-1] + imgs.shape[2:])
            cls_score = self.forward_net(imgs)
        else:
            imgs = data_batch[0:2]
            # call forward
            cls_score = self.forward_net(imgs)

        return cls_score
