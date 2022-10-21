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

import os
import random

from PIL import Image

from ..registry import PIPELINES


@PIPELINES.register()
class SamplerUCF24(object):
    """
    Sample frames id.
    NOTE: Use PIL to read image here, has diff with CV2
    Args:
        num_frames(int): The amount of frames used in a video
        frame_interval(int): Sampling rate
        valid_mode(bool): True or False.
    Returns:
        frames_idx: the index of sampled #frames.
    """
    def __init__(self,
                 num_frames=16,
                 frame_interval=1,
                 valid_mode=False):
        self.num_frames = num_frames
        self.frame_interval = frame_interval if valid_mode else random.randint(1, 2)
        self.valid_mode = valid_mode

    def _get(self, frames_idxs, img_folder, results):
        imgs = []
        for idx in frames_idxs:
            img = Image.open(
                os.path.join(img_folder, '{:05d}.jpg'.format(idx))).convert('RGB')
            imgs.append(img)
        results['imgs'] = imgs
        return results

    def _make_clip(self, im_ind, max_num):
        frame_idxs = []
        for i in reversed(range(self.num_frames)):
            # make it as a loop
            i_temp = im_ind - i * self.frame_interval
            if i_temp < 1:
                i_temp = 1
            elif i_temp > max_num:
                i_temp = max_num
            frame_idxs.append(i_temp)
        return frame_idxs

    def __call__(self, results):
        img_folder, key_frame = os.path.split(results['filename'])
        frame_len = len(os.listdir(img_folder))
        key_idx = int(key_frame[0:5])
        frame_idxs = self._make_clip(key_idx, frame_len)
        return self._get(frame_idxs, img_folder, results)
