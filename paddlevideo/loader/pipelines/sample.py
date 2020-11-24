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
        data_format =results['format']

        if data_format == "frame":
            frame_dir = results['frame_dir']
            imgs = []
            for idx in frames_idx:
                img = Image.open(os.path.join(frame_dir, results['suffix'].format(idx))).convert('RGB')
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
        for i in range(self.num_seg):
            idx = 0
            if not self.valid_mode:
                if average_dur >= self.seg_len:
                    idx = random.randint(0, average_dur - self.seg_len)
                    idx += i * average_dur
                elif average_dur >= 1:
                    idx += i * average_dur
                else:
                    idx = i
            else:
                if average_dur >= self.seg_len:
                    idx = (average_dur - 1) // 2
                    idx += i * average_dur
                elif average_dur >= 1:
                    idx += i * average_dur
                else:
                    idx = i
            for jj in range(idx, idx+self.seg_len):
                if results['format'] == 'video':
                    frames_idx.append(int(jj%frames_len))
                elif results['format'] == 'frame':
                    frames_idx.append(jj+1)
                else:
                    raise NotImplementedError

        return self._get(frames_idx, results)
