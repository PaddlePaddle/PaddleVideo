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

from .adds_head import AddsHead
from .attention_lstm_head import AttentionLstmHead
from .base import BaseHead
from .bbox_head import BBoxHeadAVA
from .pptsm_head import ppTSMHead
from .pptsn_head import ppTSNHead
from .roi_head import AVARoIHead
from .single_straight3d import SingleRoIExtractor3D
from .slowfast_head import SlowFastHead
from .stgcn_head import STGCNHead
from .timesformer_head import TimeSformerHead
from .transnetv2_head import TransNetV2Head
from .tsm_head import TSMHead
from .tsn_head import TSNHead
from .bcn_head import BcnBgmHead

__all__ = [
    'BaseHead', 'TSNHead', 'TSMHead', 'ppTSMHead', 'ppTSNHead', 'SlowFastHead',
    'AttentionLstmHead', 'TimeSformerHead', 'STGCNHead', 'TransNetV2Head',
    'SingleRoIExtractor3D', 'AVARoIHead', 'BBoxHeadAVA', 'AddsHead',
    'BcnBgmHead'
]
