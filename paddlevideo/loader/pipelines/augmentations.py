#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import random
import numpy as np
import math
from PIL import Image
from ..registry import PIPELINES
from collections.abc import Sequence


@PIPELINES.register()
class Scale(object):
    """
    Scale images.
    Args:
        short_size(float | int): Short size of an image will be scaled to the short_size.
    """
    def __init__(self, short_size):
        self.short_size = short_size

    def __call__(self, results):
        """
        Performs resize operations.
        Args:
            imgs (Sequence[PIL.Image]): List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            resized_imgs: List where each item is a PIL.Image after scaling.
        """
        imgs = results['imgs']
        resized_imgs = []
        for i in range(len(imgs)):
            img = imgs[i]
            w, h = img.size
            if (w <= h and w == self.short_size) or (h <= w
                                                     and h == self.short_size):
                resized_imgs.append(img)
                continue

            if w < h:
                ow = self.short_size
                oh = int(self.short_size * 4.0 / 3.0)
                resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
            else:
                oh = self.short_size
                ow = int(self.short_size * 4.0 / 3.0)
                resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
        results['imgs'] = resized_imgs
        return results


@PIPELINES.register()
class RandomCrop(object):
    """
    Random crop images.
    Args:
        target_size(int): Random crop a square with the target_size from an image.
    """
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, results):
        """
        Performs random crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            crop_imgs: List where each item is a PIL.Image after random crop.
        """
        imgs = results['imgs']
        w, h = imgs[0].size
        th, tw = self.target_size, self.target_size

        assert (w >= self.target_size) and (h >= self.target_size), \
            "image width({}) and height({}) should be larger than crop size".format(
                w, h, self.target_size)

        crop_images = []
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in imgs:
            if w == tw and h == th:
                crop_images.append(img)
            else:
                crop_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        results['imgs'] = crop_images
        return results


@PIPELINES.register()
class CenterCrop(object):
    """
    Center crop images.
    Args:
        target_size(int): Center crop a square with the target_size from an image.
    """
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, results):
        """
        Performs Center crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            ccrop_imgs: List where each item is a PIL.Image after Center crop.
        """
        imgs = results['imgs']
        ccrop_imgs = []
        for img in imgs:
            w, h = img.size
            th, tw = self.target_size, self.target_size
            assert (w >= self.target_size) and (h >= self.target_size), \
                "image width({}) and height({}) should be larger than crop size".format(
                    w, h, self.target_size)
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            ccrop_imgs.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        results['imgs'] = ccrop_imgs
        return results


@PIPELINES.register()
class MultiScaleCrop(object):
    def __init__(
            self,
            target_size,  #NOTE: named target size now, but still pass short size in it!
            scales=None,
            max_distort=1,
            fix_crop=True,
            more_fix_crop=True):
        self.target_size = target_size
        self.scales = scales if scales else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop

    def __call__(self, results):
        """
        Performs MultiScaleCrop operations.
        Args:
            imgs: List where wach item is a PIL.Image.
            XXX:
        results:

        """
        imgs = results['imgs']

        input_size = [self.target_size, self.target_size]

        im_size = imgs[0].size

        # get random crop offset
        def _sample_crop_size(im_size):
            image_w, image_h = im_size[0], im_size[1]

            base_size = min(image_w, image_h)
            crop_sizes = [int(base_size * x) for x in self.scales]
            crop_h = [
                input_size[1] if abs(x - input_size[1]) < 3 else x
                for x in crop_sizes
            ]
            crop_w = [
                input_size[0] if abs(x - input_size[0]) < 3 else x
                for x in crop_sizes
            ]

            pairs = []
            for i, h in enumerate(crop_h):
                for j, w in enumerate(crop_w):
                    if abs(i - j) <= self.max_distort:
                        pairs.append((w, h))
            crop_pair = random.choice(pairs)
            if not self.fix_crop:
                w_offset = random.randint(0, image_w - crop_pair[0])
                h_offset = random.randint(0, image_h - crop_pair[1])
            else:
                w_step = (image_w - crop_pair[0]) / 4
                h_step = (image_h - crop_pair[1]) / 4

                ret = list()
                ret.append((0, 0))  # upper left
                if w_step != 0:
                    ret.append((4 * w_step, 0))  # upper right
                if h_step != 0:
                    ret.append((0, 4 * h_step))  # lower left
                if h_step != 0 and w_step != 0:
                    ret.append((4 * w_step, 4 * h_step))  # lower right
                if h_step != 0 or w_step != 0:
                    ret.append((2 * w_step, 2 * h_step))  # center

                if self.more_fix_crop:
                    ret.append((0, 2 * h_step))  # center left
                    ret.append((4 * w_step, 2 * h_step))  # center right
                    ret.append((2 * w_step, 4 * h_step))  # lower center
                    ret.append((2 * w_step, 0 * h_step))  # upper center

                    ret.append((1 * w_step, 1 * h_step))  # upper left quarter
                    ret.append((3 * w_step, 1 * h_step))  # upper right quarter
                    ret.append((1 * w_step, 3 * h_step))  # lower left quarter
                    ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

                w_offset, h_offset = random.choice(ret)

            return crop_pair[0], crop_pair[1], w_offset, h_offset

        crop_w, crop_h, offset_w, offset_h = _sample_crop_size(im_size)
        crop_img_group = [
            img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in imgs
        ]
        ret_img_group = [
            img.resize((input_size[0], input_size[1]), Image.BILINEAR)
            for img in crop_img_group
        ]
        results['imgs'] = ret_img_group
        return results


@PIPELINES.register()
class RandomFlip(object):
    """
    Random Flip images.
    Args:
        p(float): Random flip images with the probability p.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, results):
        """
        Performs random flip operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            flip_imgs: List where each item is a PIL.Image after random flip.
        """
        imgs = results['imgs']
        v = random.random()
        if v < self.p:
            results['imgs'] = [
                img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs
            ]
        else:
            results['imgs'] = imgs
        return results


@PIPELINES.register()
class Image2Array(object):
    """
    transfer PIL.Image to Numpy array and transpose dimensions from 'dhwc' to 'dchw'.
    Args:
        transpose: whether to transpose or not, default False. True for tsn.
    """
    def __init__(self, transpose=True):
        self.transpose = transpose

    def __call__(self, results):
        """
        Performs Image to NumpyArray operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            np_imgs: Numpy array.
        """
        imgs = results['imgs']
        np_imgs = (np.stack(imgs)).astype('float32')
        if self.transpose:
            np_imgs = np_imgs.transpose(0, 3, 1, 2)  #nchw
        results['imgs'] = np_imgs
        return results


@PIPELINES.register()
class Normalization(object):
    """
    Normalization.
    Args:
        mean(Sequence[float]): mean values of different channels.
        std(Sequence[float]): std values of different channels.
        tensor_shape(list): size of mean, default [3,1,1]. For slowfast, [1,1,1,3]
    """
    def __init__(self, mean, std, tensor_shape=[3, 1, 1]):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}')
        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')
        self.mean = np.array(mean).reshape(tensor_shape).astype(np.float32)
        self.std = np.array(std).reshape(tensor_shape).astype(np.float32)

    def __call__(self, results):
        """
        Performs normalization operations.
        Args:
            imgs: Numpy array.
        return:
            np_imgs: Numpy array after normalization.
        """
        imgs = results['imgs']
        norm_imgs = imgs / 255.
        norm_imgs -= self.mean
        norm_imgs /= self.std
        results['imgs'] = norm_imgs
        return results


@PIPELINES.register()
class JitterScale(object):
    """
    Scale image, while the target short size is randomly select between min_size and max_size.
    Args:
        min_size: Lower bound for random sampler.
        max_size: Higher bound for random sampler.
    """
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, results):
        """
        Performs jitter resize operations.
        Args:
            imgs (Sequence[PIL.Image]): List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            resized_imgs: List where each item is a PIL.Image after scaling.
        """
        imgs = results['imgs']
        size = int(round(np.random.uniform(self.min_size, self.max_size)))
        assert (len(imgs) >= 1) , \
            "len(imgs):{} should be larger than 1".format(len(imgs))
        width, height = imgs[0].size
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
        for j in range(len(imgs)):
            img = imgs[j]
            scale_img = img.resize((new_width, new_height), Image.BILINEAR)
            frames_resize.append(scale_img)

        results['imgs'] = frames_resize
        return results


@PIPELINES.register()
class MultiCrop(object):
    """
    Random crop image.
    This operation can perform multi-crop during multi-clip test, as in slowfast model.
    Args:
        target_size(int): Random crop a square with the target_size from an image.
    """
    def __init__(self, target_size, test_mode=False):
        self.target_size = target_size
        self.test_mode = test_mode

    def __call__(self, results):
        """
        Performs random crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            crop_imgs: List where each item is a PIL.Image after random crop.
        """
        imgs = results['imgs']
        spatial_sample_index = results['spatial_sample_index']
        spatial_num_clips = results['spatial_num_clips']

        w, h = imgs[0].size
        if w == self.target_size and h == self.target_size:
            return results

        assert (w >= self.target_size) and (h >= self.target_size), \
            "image width({}) and height({}) should be larger than crop size({},{})".format(w, h, self.target_size, self.target_size)
        frames_crop = []
        if not self.test_mode:
            x_offset = random.randint(0, w - self.target_size)
            y_offset = random.randint(0, h - self.target_size)
        else:  #multi-crop
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

        for img in imgs:
            nimg = img.crop((x_offset, y_offset, x_offset + self.target_size,
                             y_offset + self.target_size))
            frames_crop.append(nimg)
        results['imgs'] = frames_crop
        return results


@PIPELINES.register()
class PackOutput(object):
    """
    In slowfast model, we want to get slow pathway from fast pathway based on
    alpha factor.
    Args:
        alpha(int): temporal length of fast/slow
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, results):
        fast_pathway = results['imgs']

        # sample num points between start and end
        slow_idx_start = 0
        slow_idx_end = fast_pathway.shape[0] - 1
        slow_idx_num = fast_pathway.shape[0] // self.alpha
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
