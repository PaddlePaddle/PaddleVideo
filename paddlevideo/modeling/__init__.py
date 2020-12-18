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
# See the License for the specific language governing permissions and
# limitations under the License.

from .backbones import ResNet
from .builder import (build_backbone, build_head, build_recognizer,
                      build_localizer, build_loss)
from .heads import BaseHead, TSNHead
from .losses import CrossEntropyLoss
from .framework.recognizers import BaseRecognizer, recognizer2d
from .registry import BACKBONES, HEADS, LOSSES, RECOGNIZERS, LOCALIZERS
from .weight_init import weight_init_

__all__ = [
    'BACKBONES',
    'HEADS',
    'RECOGNIZERS',
    'LOCALIZERS',
    'LOSSES',
    'build_recognizer',
    'build_localizer',
    'build_head',
    'build_backbone',
    'build_loss',
    'ResNet',
    'TSNHead',
    'BaseHead',
    'BaseRecognizer',
    'Recognizer2d',
    'CrossEntropyLoss',
]
