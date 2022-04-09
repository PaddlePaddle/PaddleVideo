#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import paddle
from paddle.vision.transforms import functional as F
from paddle.vision.transforms import RandomHorizontalFlip, rotate, ToTensor, Normalize, RandomCrop
import numpy as np
from ..registry import PIPELINES


@PIPELINES.register()
class FFANetDecode(object):
    """Example Pipeline """

    def __init__(self, crop_size=240, test_mode=False):
        self.crop_size = crop_size
        self.test_mode = test_mode

    def __call__(self, results):
        haze = results['haze']
        clear = results['clear']
        haze = haze.convert("RGB")
        clear = clear.convert("RGB")
        if not isinstance(self.crop_size, str):  #训练的时候对图进行裁剪
            transform = RandomCrop(self.crop_size)
            i, j, h, w = transform._get_param(haze,
                                              output_size=(self.crop_size,
                                                           self.crop_size))
            haze = np.array(haze)[i:i + h, j:j + w, :]
            clear = np.array(clear)[i:i + h, j:j + w, :]
        if not self.test_mode:  #训练的时候对图进行随机旋转和翻转
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)

            haze = RandomHorizontalFlip(rand_hor)(haze)
            clear = RandomHorizontalFlip(rand_hor)(clear)
            if rand_rot:
                haze = rotate(haze, 90 * rand_rot)
                clear = rotate(clear, 90 * rand_rot)
        haze = F.to_tensor(haze)
        results['haze'] = Normalize(mean=[0.64, 0.6, 0.58],
                                    std=[0.14, 0.15, 0.152])(haze)
        results['clear'] = F.to_tensor(clear)
        return results
