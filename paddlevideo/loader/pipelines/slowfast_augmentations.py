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

import math
import random
import numpy as np
from PIL import Image, ImageEnhance


@PIPELINES.register()
class SFScale(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, results):
        frames_select = results['imgs']
        size = int(round(np.random.uniform(self.min_size, self.max_size)))
        assert (len(frames_select) >= 1) , \
            "len(frames_select):{} should be larger than 1".format(len(frames_select))
        width, height = frames_select[0].size
        if (width <= height and width == size) or (height <= width
                                                   and height == size):
            return results

        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
        else:
            new_width = int(math.floor((float(width) / height) * size))

        frames_resize = []
        for j in range(len(frames_select)):
            img = frames_select[j]
            scale_img = img.resize((new_width, new_height), Image.BILINEAR)
            frames_resize.append(scale_img)

        results['imgs'] = frames_resize
        return results


@PIPELINES.register()
class SFCrop(object):
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, results):
        frames_resize = results['imgs']
        spatial_sample_index = results['spatial_sample_index']
        spatial_num_clips = results['spatial_num_clips']

        w, h = frames_resize[0].size
        if w == self.target_size and h == self.target_size:
            return results

        assert (w >= self.target_size) and (h >= self.target_size), \
            "image width({}) and height({}) should be larger than crop size({},{})".format(w, h, self.target_size, self.target_size)
        frames_crop = []
        if spatial_sample_index == -1:
            x_offset = random.randint(0, w - self.target_size)
            y_offset = random.randint(0, h - self.target_size)
        else:
            x_gap = int(
                math.ceil((w - self.target_size) / (spatial_num_clips - 1)))
            y_gap = int(
                math.ceil((h - self.target_size) / (spatial_num_clips - 1)))
            if h > w:
                x_offset = int(math.ceil((w - self.target_size) / 2))
                if spatial_sample_index == 0:
                    y_offset = 0
                elif spatial_sample_index == spatial_num_clips - 1:
                    y_offset = h - self.target_size
                else:
                    y_offset = y_gap * spatial_sample_index
            else:
                y_offset = int(math.ceil((h - self.target_size) / 2))
                if spatial_sample_index == 0:
                    x_offset = 0
                elif spatial_sample_index == spatial_num_clips - 1:
                    x_offset = w - self.target_size
                else:
                    x_offset = x_gap * spatial_sample_index

        for img in frames_resize:
            nimg = img.crop((x_offset, y_offset, x_offset + self.target_size,
                             y_offset + self.target_size))
            frames_crop.append(nimg)
        results['imgs'] = frames_crop
        return results


@PIPELINES.register()
class SFlip(object):
    def __init__(self):
        pass

    def __call__(self, results):
        frames_crop = results['imgs']
        spatial_sample_index = results['spatial_sample_index']
        # to move  spatial_sample_index in info to config.yaml
        # without flip when test
        if spatial_sample_index != -1:
            return results

        frames_flip = []
        if np.random.uniform() < 0.5:
            for img in frames_crop:
                nimg = img.transpose(Image.FLIP_LEFT_RIGHT)
                frames_flip.append(nimg)
        else:
            frames_flip = frames_crop

        results['imgs'] = frames_flip
        return results


@PIPELINES.register()
class SFColorNorm(object):
    def __init__(self, c_mean, c_std):
        self.c_mean = c_mean
        self.c_std = c_std

    def __call__(self, results):
        frames_flip = results['imgs']
        npframes_norm = (np.stack(frames_flip)).astype('float32')
        npframes_norm /= 255.0
        npframes_norm -= np.array(self.c_mean).reshape([1, 1, 1,
                                                        3]).astype(np.float32)
        npframes_norm /= np.array(self.c_std).reshape([1, 1, 1,
                                                       3]).astype(np.float32)
        results['imgs'] = npframes_norm
        return results


@PIPELINES.register()
class SFPackOutput(object):
    def __init__(self, slowfast_alpha):
        self.slowfast_alpha = slowfast_alpha

    def __call__(self, results):
        npframes_norm = results['imgs']
        fast_pathway = npframes_norm

        # sample num points between start and end
        slow_idx_start = 0
        slow_idx_end = fast_pathway.shape[0] - 1
        slow_idx_num = fast_pathway.shape[0] // self.slowfast_alpha
        slow_idxs_select = np.linspace(slow_idx_start, slow_idx_end,
                                       slow_idx_num).astype("int64")
        slow_pathway = fast_pathway[slow_idxs_select]

        # T H W C -> C T H W.
        slow_pathway = slow_pathway.transpose(3, 0, 1, 2)
        fast_pathway = fast_pathway.transpose(3, 0, 1, 2)

        # slow + fast
        frames_list = [slow_pathway, fast_pathway]
        results['imgs'] = frames_list
        return results
