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
from .augmentations import (CenterCrop, ColorJitter, GroupRandomFlip,
                            GroupResize, Image2Array, JitterScale, MultiCrop,
                            Normalization, PackOutput, RandomCrop, RandomFlip,
                            RandomResizedCrop, Scale, TenCrop, ToArray,
                            UniformCrop)
from .augmentations_ava import *
from .compose import Compose
from .decode import FeatureDecoder, FrameDecoder, VideoDecoder
from .decode_image import ImageDecoder
from .decode_sampler import DecodeSampler
from .mix import Cutmix, Mixup, VideoMix
from .multimodal import FeaturePadding, RandomCap, RandomMask, Tokenize
from .sample import Sampler
from .sample_ava import *
from .segmentation import MultiNorm, MultiRestrictSize
from .skeleton_pipeline import AutoPadding, Iden, SkeletonNorm

__all__ = [
    'ToArray', 'Normalization', 'LoadFeat', 'UniformCrop', 'MultiCrop',
    'FeaturePadding', 'Compose', 'RandomCap', 'GetMatchMap', 'GetVideoLabel',
    'RandomCrop', 'Image2Array', 'CenterCrop', 'VideoDecoder', 'VideoMix',
    'AutoPadding', 'RandomResizedCrop', 'FeatureDecoder', 'Scale',
    'SkeletonNorm', 'Iden', 'Mixup', 'TenCrop', 'Tokenize', 'DecodeSampler',
    'FrameDecoder', 'Sampler', 'RandomFlip', 'ImageDecoder', 'ColorJitter',
    'GroupRandomFlip', 'MultiNorm', 'RandomMask', 'PackOutput', 'JitterScale',
    'GroupResize', 'MultiRestrictSize', 'Cutmix'
]
