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

from .anet_pipeline import GetMatchMap, GetVideoLabel, LoadFeat
from .augmentations import (CenterCrop, Image2Array, JitterScale, MultiCrop,
                            Normalization, PackOutput, RandomCrop, RandomFlip,
                            Scale, TenCrop, UniformCrop)
from .compose import Compose
from .decode import FeatureDecoder, FrameDecoder, VideoDecoder
from .decode_sampler import DecodeSampler
from .mix import Cutmix, Mixup, VideoMix
from .sample import Sampler
from .skeleton_pipeline import AutoPadding, Iden, SkeletonNorm
from .augmentations_ava import *
from .sample_ava import *
from .multimodal import FeaturePadding, RandomCap, Tokenize, RandomMask
from .segmentation import MultiRestrictSize, MultiNorm

__all__ = [
    'Scale', 'RandomCrop', 'CenterCrop', 'RandomFlip', 'Image2Array',
    'Normalization', 'Compose', 'VideoDecoder', 'FrameDecoder', 'Sampler',
    'Mixup', 'Cutmix', 'JitterScale', 'MultiCrop', 'PackOutput', 'TenCrop',
    'UniformCrop', 'DecodeSampler', 'LoadFeat', 'GetMatchMap', 'GetVideoLabel',
    'AutoPadding', 'SkeletonNorm', 'Iden', 'VideoMix', 'FeatureDecoder',
    'FeaturePadding', 'RandomCap', 'Tokenize', 'RandomMask',
    'MultiRestrictSize', 'MultiNorm'
]
