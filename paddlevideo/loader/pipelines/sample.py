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
import random
from PIL import Image
from ..registry import PIPELINES
import os
import numpy as np


@PIPELINES.register()
class Sampler(object):
    """
    Sample frames id.
    NOTE: Use PIL to read image here, has diff with CV2
    Args:
        num_seg(int): number of segments.
        seg_len(int): number of sampled frames in each segment.
        mode(str): 'train', 'valid'
    Returns:
        frames_idx: the index of sampled #frames.
    """
    def __init__(self, num_seg, seg_len, valid_mode=False):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.valid_mode = valid_mode

    def _get(self, frames_idx, results):
        data_format = results['format']

        if data_format == "frame":
            frame_dir = results['frame_dir']
            imgs = []
            for idx in frames_idx:
                img = Image.open(
                    os.path.join(frame_dir,
                                 results['suffix'].format(idx))).convert('RGB')
                imgs.append(img)

        elif data_format == "video":

            frames = np.array(results['frames'])
            imgs = []
            for idx in frames_idx:
                imgbuf = frames[idx]
                img = Image.fromarray(imgbuf, mode='RGB')
                imgs.append(img)
        else:
            raise NotImplementedError
        results['imgs'] = imgs
        return results

    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        frames_len = results['frames_len']
        average_dur = int(int(frames_len) / self.num_seg)
        frames_idx = []
        if not self.valid_mode:
            if average_dur > 0:
                offsets = np.multiply(list(range(self.num_seg)),
                                      average_dur) + np.random.randint(
                                          average_dur, size=self.num_seg)
            elif int(frames_len) > self.num_seg:
                offsets = np.sort(
                    np.random.randint(int(frames_len), size=self.num_seg))
            else:
                offsets = np.zeros(shape=(self.num_seg, ))
        else:
            if int(frames_len) > self.num_seg:
                tick = (int(frames_len)) / self.num_seg
                offsets = np.array(
                    [int(tick / 2.0 + tick * x) for x in range(self.num_seg)])
            else:
                offsets = np.zeros(shape=(self.num_seg, ))

        if results['format'] == 'video':
            frames_idx = list(offsets)
            frames_idx = [x % int(frames_len) for x in frames_idx]
        elif results['format'] == 'frame':
            frames_idx = list(offsets + 1)
        else:
            raise NotImplementedError
        return self._get(frames_idx, results)
