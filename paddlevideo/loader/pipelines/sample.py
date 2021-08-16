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
import paddle


@PIPELINES.register()
class Sampler(object):
    """
    Sample frames id.
    NOTE: Use PIL to read image here, has diff with CV2
    Args:
        num_seg(int): number of segments.
        seg_len(int): number of sampled frames in each segment.
        valid_mode(bool): True or False.
        select_left: Whether to select the frame to the left in the middle when the sampling interval is even in the test mode.
    Returns:
        frames_idx: the index of sampled #frames.
    """
    def __init__(self,
                 num_seg,
                 seg_len,
                 valid_mode=False,
                 select_left=False,
                 dense_sample=False,
                 linspace_sample=False):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.valid_mode = valid_mode
        self.select_left = select_left
        self.dense_sample = dense_sample
        self.linspace_sample = linspace_sample

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
            if results['backend'] == 'cv2':
                frames = np.array(results['frames'])
                imgs = []
                for idx in frames_idx:
                    imgbuf = frames[idx]
                    img = Image.fromarray(imgbuf, mode='RGB')
                    imgs.append(img)
            elif results['backend'] == 'decord':
                vr = results['frames']
                frames_select = vr.get_batch(frames_idx)
                # dearray_to_img
                np_frames = frames_select.asnumpy()
                imgs = []
                for i in range(np_frames.shape[0]):
                    imgbuf = np_frames[i]
                    imgs.append(Image.fromarray(imgbuf, mode='RGB'))
            elif results['backend'] == 'pyav':
                imgs = []
                frames = np.array(results['frames'])
                for idx in frames_idx:
                    imgbuf = frames[idx]
                    imgs.append(imgbuf)
                imgs = np.stack(imgs)  # thwc
            else:
                raise NotImplementedError
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
        frames_len = int(results['frames_len'])
        average_dur = int(frames_len / self.num_seg)
        frames_idx = []
        if self.linspace_sample:
            if 'start_idx' in results and 'end_idx' in results:
                offsets = np.linspace(results['start_idx'], results['end_idx'], self.num_seg)
            else:
                offsets = np.linspace(0, frames_len - 1, self.num_seg)
            offsets = np.clip(offsets, 0, frames_len - 1).astype(np.long)
            if results['format'] == 'video':
                frames_idx = list(offsets)
                frames_idx = [x % frames_len for x in frames_idx]
            elif results['format'] == 'frame':
                frames_idx = list(offsets + 1)
            else:
                raise NotImplementedError
            return self._get(frames_idx, results)

        if not self.select_left:
            if self.dense_sample:  # For ppTSM
                if not self.valid_mode:  # train
                    sample_pos = max(1, 1 + frames_len - 64)
                    t_stride = 64 // self.num_seg
                    start_idx = 0 if sample_pos == 1 else np.random.randint(
                        0, sample_pos - 1)
                    offsets = [(idx * t_stride + start_idx) % frames_len + 1
                               for idx in range(self.num_seg)]
                    frames_idx = offsets
                else:
                    sample_pos = max(1, 1 + frames_len - 64)
                    t_stride = 64 // self.num_seg
                    start_list = np.linspace(0,
                                             sample_pos - 1,
                                             num=10,
                                             dtype=int)
                    offsets = []
                    for start_idx in start_list.tolist():
                        offsets += [
                            (idx * t_stride + start_idx) % frames_len + 1
                            for idx in range(self.num_seg)
                        ]
                    frames_idx = offsets
            else:
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
                    for jj in range(idx, idx + self.seg_len):
                        if results['format'] == 'video':
                            frames_idx.append(int(jj % frames_len))
                        elif results['format'] == 'frame':
                            frames_idx.append(jj + 1)
                        else:
                            raise NotImplementedError

            return self._get(frames_idx, results)

        else:  # for TSM
            if not self.valid_mode:
                if average_dur > 0:
                    offsets = np.multiply(list(range(self.num_seg)),
                                          average_dur) + np.random.randint(
                                              average_dur, size=self.num_seg)
                elif frames_len > self.num_seg:
                    offsets = np.sort(
                        np.random.randint(frames_len, size=self.num_seg))
                else:
                    offsets = np.zeros(shape=(self.num_seg, ))
            else:
                if frames_len > self.num_seg:
                    average_dur_float = frames_len / self.num_seg
                    offsets = np.array([
                        int(average_dur_float / 2.0 + average_dur_float * x)
                        for x in range(self.num_seg)
                    ])
                else:
                    offsets = np.zeros(shape=(self.num_seg, ))

            if results['format'] == 'video':
                frames_idx = list(offsets)
                frames_idx = [x % frames_len for x in frames_idx]
            elif results['format'] == 'frame':
                frames_idx = list(offsets + 1)
            else:
                raise NotImplementedError

            return self._get(frames_idx, results)
