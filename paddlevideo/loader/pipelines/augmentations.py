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
import paddle

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


#=================ava augmentation==============
import cv2
if Image is not None:
    pillow_interp_codes = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'box': Image.BOX,
        'lanczos': Image.LANCZOS,
        'hamming': Image.HAMMING
    }

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'

def _scale_size(size, scale):
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)

def rescale_size(old_size, scale, return_scale=False):
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size

def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None,
             backend=None):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = img.shape[:2]
    if backend is None:
        backend = 'cv2'
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported for resize.'
                         f"Supported backends are 'cv2', 'pillow'")

    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale

@PIPELINES.register()
class EntityBoxRescale:
    """Rescale the entity box and proposals according to the image shape.

    Required keys are "proposals", "gt_bboxes", added or modified keys are
    "gt_bboxes". If original "proposals" is not None, "proposals" and
    will be added or modified.

    Args:
        scale_factor (np.ndarray): The scale factor used entity_box rescaling.
    """

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, results):
        scale_factor = np.concatenate([self.scale_factor, self.scale_factor])

        proposals = results['proposals']
        gt_bboxes = results['gt_bboxes']
        results['gt_bboxes'] = gt_bboxes * scale_factor

        if proposals is not None:
            assert proposals.shape[1] == 4, (
                'proposals shape should be in '
                f'(n, 4), but got {proposals.shape}')
            results['proposals'] = proposals * scale_factor

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(scale_factor={self.scale_factor})'

@PIPELINES.register()
class EntityBoxCrop:
    """Crop the entity boxes and proposals according to the cropped images.

    Required keys are "proposals", "gt_bboxes", added or modified keys are
    "gt_bboxes". If original "proposals" is not None, "proposals" will be
    modified.

    Args:
        crop_bbox(np.ndarray | None): The bbox used to crop the original image.
    """

    def __init__(self, crop_bbox):
        self.crop_bbox = crop_bbox

    def __call__(self, results):
        proposals = results['proposals']
        gt_bboxes = results['gt_bboxes']

        if self.crop_bbox is None:
            return results

        x1, y1, x2, y2 = self.crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        assert gt_bboxes.shape[-1] == 4
        gt_bboxes_ = gt_bboxes.copy()
        gt_bboxes_[..., 0::2] = np.clip(gt_bboxes[..., 0::2] - x1, 0,
                                        img_w - 1)
        gt_bboxes_[..., 1::2] = np.clip(gt_bboxes[..., 1::2] - y1, 0,
                                        img_h - 1)
        results['gt_bboxes'] = gt_bboxes_

        if proposals is not None:
            assert proposals.shape[-1] == 4
            proposals_ = proposals.copy()
            proposals_[..., 0::2] = np.clip(proposals[..., 0::2] - x1, 0,
                                            img_w - 1)
            proposals_[..., 1::2] = np.clip(proposals[..., 1::2] - y1, 0,
                                            img_h - 1)
            results['proposals'] = proposals_
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(crop_bbox={self.crop_bbox})'

@PIPELINES.register()
class EntityBoxFlip:
    """Flip the entity boxes and proposals with a probability.

    Reverse the order of elements in the given bounding boxes and proposals
    with a specific direction. The shape of them are preserved, but the
    elements are reordered. Only the horizontal flip is supported (seems
    vertical flipping makes no sense). Required keys are "proposals",
    "gt_bboxes", added or modified keys are "gt_bboxes". If "proposals"
    is not None, it will also be modified.

    Args:
        img_shape (tuple[int]): The img shape.
    """

    def __init__(self, img_shape):
        self.img_shape = img_shape
        # assert mmcv.is_tuple_of(img_shape, int)

    def __call__(self, results):
        proposals = results['proposals']
        gt_bboxes = results['gt_bboxes']
        img_h, img_w = self.img_shape

        assert gt_bboxes.shape[-1] == 4
        gt_bboxes_ = gt_bboxes.copy()
        gt_bboxes_[..., 0::4] = img_w - gt_bboxes[..., 2::4] - 1
        gt_bboxes_[..., 2::4] = img_w - gt_bboxes[..., 0::4] - 1
        if proposals is not None:
            assert proposals.shape[-1] == 4
            proposals_ = proposals.copy()
            proposals_[..., 0::4] = img_w - proposals[..., 2::4] - 1
            proposals_[..., 2::4] = img_w - proposals[..., 0::4] - 1
        else:
            proposals_ = None

        results['proposals'] = proposals_
        results['gt_bboxes'] = gt_bboxes_
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(img_shape={self.img_shape})'
        return repr_str


@PIPELINES.register()
class Resize:
    """Resize images to a specific size.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "lazy",
    "resize_size". Required keys in "lazy" is None, added or modified key is
    "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 lazy=False):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)

        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            results['imgs'] = [
                imresize(
                    img, (new_w, new_h), interpolation=self.interpolation)
                for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        if 'gt_bboxes' in results:
            assert not self.lazy
            entity_box_rescale = EntityBoxRescale(self.scale_factor)
            results = entity_box_rescale(results)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str

@PIPELINES.register()
class RandomRescale:
    """Randomly resize images so that the short_edge is resized to a specific
    size in a given range. The scale ratio is unchanged after resizing.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "resize_size",
    "short_edge".

    Args:
        scale_range (tuple[int]): The range of short edge length. A closed
            interval.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self, scale_range, interpolation='bilinear'):
        scale_range = eval(scale_range)
        self.scale_range = scale_range
        # make sure scale_range is legal, first make sure the type is OK
        # assert mmcv.is_tuple_of(scale_range, int)
        
        assert len(scale_range) == 2
        assert scale_range[0] < scale_range[1]
        assert np.all([x > 0 for x in scale_range])

        self.keep_ratio = True
        self.interpolation = interpolation

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        short_edge = np.random.randint(self.scale_range[0],
                                       self.scale_range[1] + 1)
        resize = Resize((-1, short_edge),
                        keep_ratio=True,
                        interpolation=self.interpolation,
                        lazy=False)
        results = resize(results)

        results['short_edge'] = short_edge
        return results

    def __repr__(self):
        scale_range = self.scale_range
        repr_str = (f'{self.__class__.__name__}('
                    f'scale_range=({scale_range[0]}, {scale_range[1]}), '
                    f'interpolation={self.interpolation})')
        return repr_str

@PIPELINES.register()
class RandomCrop_v2:
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "imgs" and "img_shape", added or
    modified keys are "imgs", "lazy"; Required keys in "lazy" are "flip",
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    def __call__(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = self.size / img_w, self.size / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_x_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        new_h, new_w = self.size, self.size

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])

        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            results['imgs'] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        # Process entity boxes
        if 'gt_bboxes' in results:
            assert not self.lazy
            entity_box_crop = EntityBoxCrop(results['crop_bbox'])
            results = entity_box_crop(results)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str

def imflip_(img, direction='horizontal'):
    """Inplace flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image (inplace).
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return cv2.flip(img, 1, img)
    elif direction == 'vertical':
        return cv2.flip(img, 0, img)
    else:
        return cv2.flip(img, -1, img)

def iminvert(img):
    """Invert (negate) an image.

    Args:
        img (ndarray): Image to be inverted.

    Returns:
        ndarray: The inverted image.
    """
    return np.full_like(img, 255) - img

@PIPELINES.register()
class Flip:
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.
    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "lazy" and "flip_direction". Required keys in "lazy" is
    None, added or modified key are "flip" and "flip_direction". The Flip
    augmentation should be placed after any cropping / reshaping augmentations,
    to make sure crop_quadruple is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self, flip_ratio=0.5, direction='horizontal', lazy=False):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.lazy = lazy

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        #modality = results['modality']
        #if modality == 'Flow':
        #    assert self.direction == 'horizontal'

        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction

        if not self.lazy:
            if flip:
                for i, img in enumerate(results['imgs']):
                    imflip_(img, self.direction)
                lt = len(results['imgs'])
                #for i in range(0, lt, 2):
                    # flow with even indexes are x_flow, which need to be
                    # inverted when doing horizontal flip
                #    if modality == 'Flow':
                #        results['imgs'][i] = iminvert(results['imgs'][i])

            else:
                results['imgs'] = list(results['imgs'])
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Use one Flip please')
            lazyop['flip'] = flip
            lazyop['flip_direction'] = self.direction

        if 'gt_bboxes' in results and flip:
            assert not self.lazy and self.direction == 'horizontal'
            entity_box_flip = EntityBoxFlip(results['img_shape'])
            results = entity_box_flip(results)

        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'lazy={self.lazy})')
        return repr_str

def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img

@PIPELINES.register()
class Normalize:
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
            )

        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr
        self.adjust_magnitude = adjust_magnitude

    def __call__(self, results):
        #modality = results['modality']

        if modality == 'RGB':
            n = len(results['imgs'])
            h, w, c = results['imgs'][0].shape
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results['imgs']):
                imgs[i] = img

            for img in imgs:
                imnormalize_(img, self.mean, self.std, self.to_bgr)

            results['imgs'] = imgs
            results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_bgr=self.to_bgr)
            return results
        if modality == 'Flow':
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            assert self.mean.shape[0] == 2
            assert self.std.shape[0] == 2
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            x_flow = (x_flow - self.mean[0]) / self.std[0]
            y_flow = (y_flow - self.mean[1]) / self.std[1]
            if self.adjust_magnitude:
                x_flow = x_flow * results['scale_factor'][0]
                y_flow = y_flow * results['scale_factor'][1]
            imgs = np.stack([x_flow, y_flow], axis=-1)
            results['imgs'] = imgs
            args = dict(
                mean=self.mean,
                std=self.std,
                to_bgr=self.to_bgr,
                adjust_magnitude=self.adjust_magnitude)
            results['img_norm_cfg'] = args
            return results
        raise NotImplementedError

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_bgr={self.to_bgr}, '
                    f'adjust_magnitude={self.adjust_magnitude})')
        return repr_str


@PIPELINES.register()
class FormatShape:
    """Format final imgs shape to the given input_format.

    Required keys are "imgs", "num_clips" and "clip_len", added or modified
    keys are "imgs" and "input_shape".

    Args:
        input_format (str): Define the final imgs format.
        collapse (bool): To collpase input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Default: False.
    """

    def __init__(self, input_format, collapse=False):
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in ['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def __call__(self, results):
        """Performs the FormatShape formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        imgs = results['imgs']
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * L
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            num_clips = results['num_clips']
            clip_len = results['clip_len']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x L x H x W x C
            imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
            # N_crops x N_clips x C x L x H x W
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # M' x C x L x H x W
            # M' = N_crops x N_clips
        elif self.input_format == 'NCHW':
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            # M x C x H x W
        elif self.input_format == 'NCHW_Flow':
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x L x H x W x C
            imgs = np.transpose(imgs, (0, 1, 2, 5, 3, 4))
            # N_crops x N_clips x L x C x H x W
            imgs = imgs.reshape((-1, imgs.shape[2] * imgs.shape[3]) +
                                imgs.shape[4:])
            # M' x C' x H x W
            # M' = N_crops x N_clips
            # C' = L x C
        elif self.input_format == 'NPTCHW':
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x L
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W
        if self.collapse:
            assert imgs.shape[0] == 1
            imgs = imgs.squeeze(0)

        results['imgs'] = imgs
        results['input_shape'] = imgs.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str

@PIPELINES.register()
class Rename:
    """Rename the key in results.

    Args:
        mapping (dict): The keys in results that need to be renamed. The key of
            the dict is the original name, while the value is the new name. If
            the original name not found in results, do nothing.
            Default: dict().
    """

    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, results):
        for key, value in self.mapping.items():
            if key in results:
                assert isinstance(key, str) and isinstance(value, str)
                assert value not in results, ('the new name already exists in '
                                              'results')
                results[value] = results[key]
                results.pop(key)
        return results

def to_tensor(data):
    import paddle
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, paddle.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return paddle.from_numpy(data)
    # if isinstance(data, Sequence) and not mmcv.is_str(data):
    #     return torch.tensor(data)
    # if isinstance(data, int):
    #     return paddle.LongTensor([data])
    # if isinstance(data, float):
    #     return torch.FloatTensor([data])
    raise TypeError(f'type {type(data)} cannot be converted to tensor.')

@PIPELINES.register()
class ToTensor:
    """Convert some values in results dict to `torch.Tensor` type in data
    loader pipeline.

    Args:
        keys (Sequence[str]): Required keys to be converted.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Performs the ToTensor formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'


class DataContainer:
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self,
                 data,
                 stack=False,
                 padding_value=0,
                 cpu_only=False,
                 pad_dims=2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.data)})'

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, paddle.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def pad_dims(self):
        return self._pad_dims

    # @assert_tensor_type
    # def size(self, *args, **kwargs):
    #     return self.data.size(*args, **kwargs)
    #
    # @assert_tensor_type
    # def dim(self):
    #     return self.data.dim()

@PIPELINES.register()
class ToDataContainer:
    """Convert the data to DataContainer.

    Args:
        fields (Sequence[dict]): Required fields to be converted
            with keys and attributes. E.g.
            fields=(dict(key='gt_bbox', stack=False),).
            Note that key can also be a list of keys, if so, every tensor in
            the list will be converted to DataContainer.
    """

    def __init__(self, fields):
        self.fields = fields

    def __call__(self, results):
        """Performs the ToDataContainer formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        for field in self.fields:
            _field = field.copy()
            key = _field.pop('key')
            if isinstance(key, list):
                for item in key:
                    results[item] = DataContainer(results[item], **_field)
            else:
                results[key] = DataContainer(results[key], **_field)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(fields={self.fields})'

@PIPELINES.register()
class Collect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in ``keys`` as it is, and collect items in
    ``meta_keys`` into a meta item called ``meta_name``.This is usually
    the last stage of the data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta infomation.
            This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
            The contents of the ``meta_name`` dictionary depends on
            ``meta_keys``.
            By default this includes:

            - "filename": path to the image file
            - "label": label of the image file
            - "original_shape": original shape of the image as a tuple
                (h, w, c)
            - "img_shape": shape of the image input to the network as a tuple
                (h, w, c).  Note that images may be zero padded on the
                bottom/right, if the batch tensor is larger than this shape.
            - "pad_shape": image shape after padding
            - "flip_direction": a str in ("horiziontal", "vertival") to
                indicate if the image is fliped horizontally or vertically.
            - "img_norm_cfg": a dict of normalization information:
                - mean - per channel mean subtraction
                - std - per channel std divisor
                - to_rgb - bool indicating if bgr was converted to rgb
        nested (bool): If set as True, will apply data[x] = [data[x]] to all
            items in data. The arg is added for compatibility. Default: False.
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'label', 'original_shape', 'img_shape',
                            'pad_shape', 'flip_direction', 'img_norm_cfg'),
                 meta_name='img_metas',
                 nested=False):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name
        self.nested = nested

    def __call__(self, results):
        """Performs the Collect formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        data = {}
        for key in self.keys:
            data[key] = results[key]

        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                meta[key] = results[key]
            data[self.meta_name] = DataContainer(meta, cpu_only=True)
        if self.nested:
            for k in data:
                data[k] = [data[k]]

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys}, '
                f'nested={self.nested})')


