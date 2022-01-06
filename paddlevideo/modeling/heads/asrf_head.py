# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.

# https://github.com/yiskw713/asrf/libs/models/tcn.py
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle import ParamAttr

from ..backbones.ms_tcn import SingleStageModel

from .base import BaseHead
from ..registry import HEADS
from ..weight_init import weight_init_


@HEADS.register()
class ASRFHead(BaseHead):

    def __init__(self,
                 num_classes,
                 num_features,
                 num_stages,
                 num_layers,
                 num_stages_asb=None,
                 num_stages_brb=None):
        super().__init__(num_classes=num_classes, in_channels=num_features)
        if not isinstance(num_stages_asb, int):
            num_stages_asb = num_stages
        
        if not isinstance(num_stages_brb, int):
            num_stages_brb = num_stages

        self.num_layers = num_layers
        self.num_stages_asb = num_stages_asb
        self.num_stages_brb = num_stages_brb
        self.num_features = num_features

        self.init_weights()
    
    def init_weights(self):
        """
        initialize model layers' weight
        """
        self.conv_cls = nn.Conv1D(self.num_features, self.num_classes, 1)
        self.conv_boundary = nn.Conv1D(self.num_features, 1, 1)

        # action segmentation branch
        asb = [
            SingleStageModel(self.num_layers, self.num_features, self.num_classes, self.num_classes)
            for _ in range(self.num_stages_asb - 1)
        ]

        # boundary regression branch
        brb=[
            SingleStageModel(self.num_layers, self.num_features, 1, 1)
            for _ in range(self.num_stages_brb - 1)
        ]
        self.brb = nn.LayerList(brb)
        self.asb = nn.LayerList(asb)

        self.activation_asb = nn.Softmax(axis=1)
        self.activation_brb = nn.Sigmoid()

        # init weight
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv1D):
                weight_init_(layer, 'XavierNormal')
                # weight_init_(layer, 'Normal', mean=0.0, std=0.02)

    def forward(self, x):
        """
        ASRF head
        """
        out_cls = self.conv_cls(x)
        out_boundary = self.conv_boundary(x)

        outputs_cls = [out_cls]
        outputs_boundary = [out_boundary]

        for as_stage in self.asb:
            out_cls = as_stage(self.activation_asb(out_cls))
            outputs_cls.append(out_cls)

        for br_stage in self.brb:
            out_boundary = br_stage(self.activation_brb(out_boundary))
            outputs_boundary.append(out_boundary)

        return outputs_cls, outputs_boundary
