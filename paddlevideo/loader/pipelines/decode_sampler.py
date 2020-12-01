# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..registry import PIPELINES

import decord as de
import random
import numpy as np
from PIL import Image


@PIPELINES.register()
class DecodeSampler(object):
    """
    Decode and sample
    Args:
        filepath: the file path of mp4 file
    """
    def __init__(self, num_frames, sampling_rate, target_fps):
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.target_fps = target_fps

    def get_start_end_idx(self, video_size, clip_size, clip_idx,
                          temporal_num_clips):
        delta = max(video_size - clip_size, 0)
        if clip_idx == -1:  # when test, temporal_num_clips is not used
            # Random temporal sampling.
            start_idx = random.uniform(0, delta)
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / temporal_num_clips
        end_idx = start_idx + clip_size - 1
        return start_idx, end_idx

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        filepath = results['filename']
        temporal_sample_index = results['temporal_sample_index']
        temporal_num_clips = results['temporal_num_clips']

        vr = de.VideoReader(filepath)
        videolen = len(vr)

        fps = vr.get_avg_fps()
        clip_size = self.num_frames * self.sampling_rate * fps / self.target_fps

        start_idx, end_idx = self.get_start_end_idx(videolen, clip_size,
                                                    temporal_sample_index,
                                                    temporal_num_clips)
        index = np.linspace(start_idx, end_idx, self.num_frames).astype("int64")
        index = np.clip(index, 0, videolen)

        frames_select = vr.get_batch(index)  #1 for buffer

        # dearray_to_img
        np_frames = frames_select.asnumpy()
        frames_select_list = []
        for i in range(np_frames.shape[0]):
            imgbuf = np_frames[i]
            frames_select_list.append(Image.fromarray(imgbuf, mode='RGB'))
        results['imgs'] = frames_select_list
        return results
