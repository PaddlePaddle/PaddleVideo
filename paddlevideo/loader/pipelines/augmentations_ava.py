#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

chaj_rand = 1 #0
chaj_debug = 0 
import random
import numpy as np
import math
from PIL import Image
from ..registry import PIPELINES
from collections.abc import Sequence
import cv2

#if Image is not None:
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

    #chajchaj 0506: aliged with mmaction, for one img_key's train pipeline
    def __call__(self, results):
        scale_factor = np.concatenate([self.scale_factor, self.scale_factor])

        proposals = results['proposals']
        gt_bboxes = results['gt_bboxes']
        #print('-----639----',gt_bboxes)
        if chaj_debug:
            print('----EntityBoxRescale, scale_factor----',scale_factor)
            print('----EntityBoxRescale, img_key:',results['img_key'],'gt_bboxes bf scale_factor:',results['gt_bboxes'])
        results['gt_bboxes'] = gt_bboxes * scale_factor
        if chaj_debug:
            print('----EntityBoxRescale, img_key:',results['img_key'],'gt_bboxes af scale_factor:',results['gt_bboxes'])

        if proposals is not None:
            assert proposals.shape[1] == 4, (
                'proposals shape should be in '
                f'(n, 4), but got {proposals.shape}')
            if chaj_debug:
                print('---EntityBoxRescale, img_key:',results['img_key'],'proposals bf scale_factor:',results['proposals'])
            results['proposals'] = proposals * scale_factor
            if chaj_debug:
                print('---EntityBoxRescale, img_key:',results['img_key'],'proposals af scale_factor:',results['proposals'])

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
        if chaj_debug:
            print('----RandomCrop_v2, EntityBoxCrop,  img_key:',results['img_key'],'gt_bboxes bf crop:',results['gt_bboxes'])
        gt_bboxes_ = gt_bboxes.copy()
        gt_bboxes_[..., 0::2] = np.clip(gt_bboxes[..., 0::2] - x1, 0,
                                        img_w - 1)
        gt_bboxes_[..., 1::2] = np.clip(gt_bboxes[..., 1::2] - y1, 0,
                                        img_h - 1)
        results['gt_bboxes'] = gt_bboxes_
        if chaj_debug:
            print('----RandomCrop_v2, EntityBoxCrop,  img_key:',results['img_key'],'gt_bboxes af crop:',results['gt_bboxes'])
        #print('-----687---- A BoxCrop',results['gt_bboxes']) 

        if proposals is not None:
            assert proposals.shape[-1] == 4
            if chaj_debug:
                print('----RandomCrop_v2, EntityBoxCrop,  img_key:',results['img_key'],'proposals bf crop:',results['proposals'])
            proposals_ = proposals.copy()
            proposals_[..., 0::2] = np.clip(proposals[..., 0::2] - x1, 0,
                                            img_w - 1)
            proposals_[..., 1::2] = np.clip(proposals[..., 1::2] - y1, 0,
                                            img_h - 1)
            results['proposals'] = proposals_
            if chaj_debug:
                print('----RandomCrop_v2, EntityBoxCrop,  img_key:',results['img_key'],'proposals bf crop:',results['proposals'])
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
        if chaj_debug:
            print("chajchaj, Flip, img_key:",results['img_key'], "bf flip, results['proposals']:",results['proposals'])
            print("chajchaj, Flip, img_key:",results['img_key'], "bf flip, results['gt_bboxes']:",results['gt_bboxes'])

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
        if chaj_debug:
            print("chajchaj, Flip, img_key:",results['img_key'], "af flip, results['proposals']:",results['proposals'])
            print("chajchaj, Flip, img_key:",results['img_key'], "af flip, results['gt_bboxes']:",results['gt_bboxes'])

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
        #scale = tuple(scale)
        #print("chajchaj,augmentations.py,Scale, scale:",scale) 
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
        #print('----813---scale_factor in Resize',self.scale_factor)
        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        
        if not self.lazy: #chajchaj 0506: aliged with mmaction, for one img_key's train pipeline
            if chaj_debug:
                print("chajchaj, augmentations.py:Resize, img_key:",results['img_key'],"bf imresize, results['imgs']:", results['imgs'])
            results['imgs'] = [
                imresize(
                    img, (new_w, new_h), interpolation=self.interpolation)
                for img in results['imgs']
            ]
            if chaj_debug:
                print("chajchaj, augmentations.py:Resize, img_key:",results['img_key'],"af imresize, results['imgs']:", results['imgs'])
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        if 'gt_bboxes' in results:
            assert not self.lazy
            #print('---- 829 in Resize ---',results['gt_bboxes'])
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
        #print('-----877---- before randomscale ------',results['gt_bboxes'])
        #chajchaj, self.scale_range[0]: 256 self.scale_range[1]: 320 self.interpolation: bilinear
        if chaj_debug:
            print("chajchaj, self.scale_range[0]:",self.scale_range[0],"self.scale_range[1]:",self.scale_range[1],"self.interpolation:",self.interpolation)
        if not chaj_rand: #去除随机逻辑
            short_edge = 300
        else:
            short_edge = np.random.randint(self.scale_range[0],
                                       self.scale_range[1] + 1)
        #print('----885 b randomscale results[proposals]----',results['proposals'])
        #print('---884 short_edge---',short_edge) 
        resize = Resize((-1, short_edge),
                        keep_ratio=True,
                        interpolation=self.interpolation,
                        lazy=False)
        results = resize(results)

        results['short_edge'] = short_edge
        #print('-----886----A Randomrescale-----',results['proposals'])
        return results

    def __repr__(self):
        scale_range = self.scale_range
        repr_str = (f'{self.__class__.__name__}('
                    f'scale_range=({scale_range[0]}, {scale_range[1]}), '
                    f'interpolation={self.interpolation})')
        return repr_str

@PIPELINES.register()
class Rescale:
    """resize images so that the short_edge is resized to a specific
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
        print("chajchaj,augmentations.py,Rescale, scale_range:",scale_range) 
        self.scale_range = scale_range
        # make sure scale_range is legal, first make sure the type is OK
        # assert mmcv.is_tuple_of(scale_range, int)
        
        #assert len(scale_range) == 2
        #assert scale_range[0] < scale_range[1]
        #assert np.all([x > 0 for x in scale_range])

        self.keep_ratio = True
        self.interpolation = interpolation

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        #print('-----877---- before randomscale ------',results['gt_bboxes'])
        #chajchaj, self.scale_range[0]: 256 self.scale_range[1]: 320 self.interpolation: bilinear
        #if chaj_debug:
        #    print("chajchaj, self.scale_range[0]:",self.scale_range[0],"self.scale_range[1]:",self.scale_range[1],"self.interpolation:",self.interpolation)
        #if not chaj_rand: #去除随机逻辑
        #    short_edge = 300
        #else:
        #    short_edge = np.random.randint(self.scale_range[0],
        #                               self.scale_range[1] + 1)
        #print('----885 b randomscale results[proposals]----',results['proposals'])
        #print('---884 short_edge---',short_edge) 
        resize = Resize(self.scale_range, #(-1, short_edge),
                        keep_ratio=True,
                        interpolation=self.interpolation,
                        lazy=False)
        results = resize(results)

        #results['short_edge'] = short_edge
        #print('-----886----A Randomrescale-----',results['proposals'])
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

        if chaj_debug:
            print("chajchaj, RandomCrop_v2, self.size:",self.size,"self.lazy:",self.lazy,"img_h:",img_h,"img_w:",img_w)
        y_offset = 0
        x_offset = 0
        if chaj_rand: #随机
            if img_h > self.size:
                y_offset = int(np.random.randint(0, img_h - self.size))
            if img_w > self.size:
                x_offset = int(np.random.randint(0, img_w - self.size))
        #print('---941  x_offset y_offset ---',(x_offset,y_offset))
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
            new_crop_quadruple, dtype=np.float32) #TODO: 这个作用是?
        if chaj_debug:
            print("chajchaj, RandomCrop_v2, results['crop_quadruple']:",results['crop_quadruple'])

        new_h, new_w = self.size, self.size

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        #print('---965  results[crop_bbox] ---',results['crop_bbox'])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if chaj_debug:
                print('----RandomCrop_v2, img_key:',results['img_key'],'imgs bf crop:',results['imgs'])
            results['imgs'] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results['imgs']
            ]
            if chaj_debug:
                print('----RandomCrop_v2, img_key:',results['img_key'],'imgs af crop:',results['imgs'])
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
            #print('---992--- b EntiBoxCrop',results['gt_bboxes'])
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
        #print(np.random.rand())
        #flip = 0.3 < self.flip_ratio
        if chaj_rand:#随机
            flip = np.random.rand() < self.flip_ratio
        else:
            flip = 1
        #print("chajchaj, Flip, img_key:",results['img_key'], "flip:",flip, "self.direction:",self.direction, "self.flip_label_map:",self.flip_label_map, "modality:",modality)
        if chaj_debug:
            print("chajchaj, Flip, img_key:",results['img_key'], "flip:",flip, "self.direction:",self.direction)

        results['flip'] = flip
        results['flip_direction'] = self.direction

        if not self.lazy:
            if flip:
                if chaj_debug:
                    print("chajchaj, Flip, img_key:",results['img_key'], "bf imflip_, results['imgs']:",results['imgs'])
                for i, img in enumerate(results['imgs']):
                    imflip_(img, self.direction)
                if chaj_debug:
                    print("chajchaj, Flip, img_key:",results['img_key'], "af imflip_, results['imgs']:",results['imgs'])
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

        # if modality == 'RGB':
        if chaj_debug:
            print('----Normalize, img_key:',results['img_key'],'imgs bf imnormalize_:',results['imgs'])
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
        
        if chaj_debug:
            print('----Normalize, img_key:',results['img_key'],'imgs af imnormalize_:',results['imgs'])
        return results
        # if modality == 'Flow':
        #     num_imgs = len(results['imgs'])
        #     assert num_imgs % 2 == 0
        #     assert self.mean.shape[0] == 2
        #     assert self.std.shape[0] == 2
        #     n = num_imgs // 2
        #     h, w = results['imgs'][0].shape
        #     x_flow = np.empty((n, h, w), dtype=np.float32)
        #     y_flow = np.empty((n, h, w), dtype=np.float32)
        #     for i in range(n):
        #         x_flow[i] = results['imgs'][2 * i]
        #         y_flow[i] = results['imgs'][2 * i + 1]
        #     x_flow = (x_flow - self.mean[0]) / self.std[0]
        #     y_flow = (y_flow - self.mean[1]) / self.std[1]
        #     if self.adjust_magnitude:
        #         x_flow = x_flow * results['scale_factor'][0]
        #         y_flow = y_flow * results['scale_factor'][1]
        #     imgs = np.stack([x_flow, y_flow], axis=-1)
        #     results['imgs'] = imgs
        #     args = dict(
        #         mean=self.mean,
        #         std=self.std,
        #         to_bgr=self.to_bgr,
        #         adjust_magnitude=self.adjust_magnitude)
        #     results['img_norm_cfg'] = args
        #     return results
        raise NotImplementedError

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_bgr={self.to_bgr}, '
                    f'adjust_magnitude={self.adjust_magnitude})')
        return repr_str


#@PIPELINES.register()
#class FormatShape:
#    """Format final imgs shape to the given input_format.
#
#    Required keys are "imgs", "num_clips" and "clip_len", added or modified
#    keys are "imgs" and "input_shape".
#
#    Args:
#        input_format (str): Define the final imgs format.
#        collapse (bool): To collpase input_format N... to ... (NCTHW to CTHW,
#            etc.) if N is 1. Should be set as True when training and testing
#            detectors. Default: False.
#    """
#
#    def __init__(self, input_format, collapse=False):
#        self.input_format = input_format
#        self.collapse = collapse
#        if self.input_format not in ['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']:
#            raise ValueError(
#                f'The input format {self.input_format} is invalid.')
#
#    def __call__(self, results):
#        """Performs the FormatShape formating.
#
#        Args:
#            results (dict): The resulting dict to be modified and passed
#                to the next transform in pipeline.
#        """
#        #chajchaj, formating.py:FormatShape, img_key: -5KQ66BBWC4,0902 self.collapse: True self.input_format: NCTHW
#        if chaj_debug:
#            print("chajchaj, formating.py:FormatShape, img_key:",results['img_key'],"self.collapse:",self.collapse, "self.input_format:",self.input_format)
#            print("chajchaj, formating.py:FormatShape, img_key:",results['img_key'],"bf reshape, results['imgs'].shape:", results['imgs'].shape)
#            print("chajchaj, formating.py:FormatShape, img_key:",results['img_key'],"bf reshape, results['imgs']:", results['imgs'])
#        imgs = results['imgs']
#        # [M x H x W x C]
#        # M = 1 * N_crops * N_clips * L
#        if self.collapse:
#            assert results['num_clips'] == 1
#
#        if self.input_format == 'NCTHW': #(32, 256, 256, 3) -> (3, 32, 256, 256)
#            #'clip_len': 32, 'num_clips': 1
#            num_clips = results['num_clips']
#            clip_len = results['clip_len']
#
#            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
#            # N_crops x N_clips x L x H x W x C
#            imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
#            # N_crops x N_clips x C x L x H x W
#            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
#            # M' x C x L x H x W
#            # M' = N_crops x N_clips
#        elif self.input_format == 'NCHW':
#            imgs = np.transpose(imgs, (0, 3, 1, 2))
#            # M x C x H x W
#        elif self.input_format == 'NCHW_Flow':
#            num_clips = results['num_clips']
#            clip_len = results['clip_len']
#            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
#            # N_crops x N_clips x L x H x W x C
#            imgs = np.transpose(imgs, (0, 1, 2, 5, 3, 4))
#            # N_crops x N_clips x L x C x H x W
#            imgs = imgs.reshape((-1, imgs.shape[2] * imgs.shape[3]) +
#                                imgs.shape[4:])
#            # M' x C' x H x W
#            # M' = N_crops x N_clips
#            # C' = L x C
#        elif self.input_format == 'NPTCHW':
#            num_proposals = results['num_proposals']
#            num_clips = results['num_clips']
#            clip_len = results['clip_len']
#            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
#                                imgs.shape[1:])
#            # P x M x H x W x C
#            # M = N_clips x L
#            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
#            # P x M x C x H x W
#        if self.collapse:
#            assert imgs.shape[0] == 1
#            imgs = imgs.squeeze(0)
#
#        results['imgs'] = imgs
#        results['input_shape'] = imgs.shape
#        if chaj_debug:
#            print("chajchaj, formating.py:FormatShape, img_key:",results['img_key'],"af reshape, results['imgs'].shape:", results['imgs'].shape)
#            print("chajchaj, formating.py:FormatShape, img_key:",results['img_key'],"af reshape, results['imgs']:", results['imgs'])
#
#        return results
#
#    def __repr__(self):
#        repr_str = self.__class__.__name__
#        repr_str += f"(input_format='{self.input_format}')"
#        return repr_str
#
#@PIPELINES.register()
#class Rename:
#    """Rename the key in results.
#
#    Args:
#        mapping (dict): The keys in results that need to be renamed. The key of
#            the dict is the original name, while the value is the new name. If
#            the original name not found in results, do nothing.
#            Default: dict().
#    """
#
#    def __init__(self, mapping):
#        self.mapping = mapping
#
#    def __call__(self, results):
#        for key, value in self.mapping.items():
#            if key in results:
#                assert isinstance(key, str) and isinstance(value, str)
#                assert value not in results, ('the new name already exists in '
#                                              'results')
#                results[value] = results[key]
#                results.pop(key)
#        return results
#
#def to_tensor(data):
#    """Convert objects of various python types to :obj:`torch.Tensor`.
#
#    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
#    :class:`Sequence`, :class:`int` and :class:`float`.
#    """
#    if isinstance(data, paddle.Tensor):
#        return data
#    if isinstance(data, np.ndarray):
#        #return paddle.to_tensor(data)
#        return paddle.to_tensor(data, place=CUDAPlace)
#    # if isinstance(data, Sequence) and not mmcv.is_str(data):
#    #     return torch.tensor(data)
#    # if isinstance(data, int):
#    #     return paddle.LongTensor([data])
#    # if isinstance(data, float):
#    #     return torch.FloatTensor([data])
#    raise TypeError(f'type {type(data)} cannot be converted to tensor.')

#1012 by chajchaj## @PIPELINES.register()
#1012 by chajchaj## class ToTensor:
#1012 by chajchaj##     """Convert some values in results dict to `torch.Tensor` type in data
#1012 by chajchaj##     loader pipeline.
#1012 by chajchaj## 
#1012 by chajchaj##     Args:
#1012 by chajchaj##         keys (Sequence[str]): Required keys to be converted.
#1012 by chajchaj##     """
#1012 by chajchaj## 
#1012 by chajchaj##     def __init__(self, keys):
#1012 by chajchaj##         self.keys = keys
#1012 by chajchaj## 
#1012 by chajchaj##     def __call__(self, results):
#1012 by chajchaj##         """Performs the ToTensor formating.
#1012 by chajchaj## 
#1012 by chajchaj##         Args:
#1012 by chajchaj##             results (dict): The resulting dict to be modified and passed
#1012 by chajchaj##                 to the next transform in pipeline.
#1012 by chajchaj##         """
#1012 by chajchaj##         for key in self.keys:
#1012 by chajchaj##             if chaj_debug:
#1012 by chajchaj##                 print("chajchaj, key:",key, "type:",type(results[key]))
#1012 by chajchaj##             results[key] = to_tensor(results[key])
#1012 by chajchaj##             if chaj_debug:
#1012 by chajchaj##                 print("chajchaj, key:",key, "af to_tensor, results[key]:",results[key] )
#1012 by chajchaj##         return results
#1012 by chajchaj## 
#1012 by chajchaj##     def __repr__(self):
#1012 by chajchaj##         return f'{self.__class__.__name__}(keys={self.keys})'
#1012 by chajchaj## 
#1012 by chajchaj## 
#1012 by chajchaj## class DataContainer:
#1012 by chajchaj##     """A container for any type of objects.
#1012 by chajchaj## 
#1012 by chajchaj##     Typically tensors will be stacked in the collate function and sliced along
#1012 by chajchaj##     some dimension in the scatter function. This behavior has some limitations.
#1012 by chajchaj##     1. All tensors have to be the same size.
#1012 by chajchaj##     2. Types are limited (numpy array or Tensor).
#1012 by chajchaj## 
#1012 by chajchaj##     We design `DataContainer` and `MMDataParallel` to overcome these
#1012 by chajchaj##     limitations. The behavior can be either of the following.
#1012 by chajchaj## 
#1012 by chajchaj##     - copy to GPU, pad all tensors to the same size and stack them
#1012 by chajchaj##     - copy to GPU without stacking
#1012 by chajchaj##     - leave the objects as is and pass it to the model
#1012 by chajchaj##     - pad_dims specifies the number of last few dimensions to do padding
#1012 by chajchaj##     """
#1012 by chajchaj## 
#1012 by chajchaj##     def __init__(self,
#1012 by chajchaj##                  data,
#1012 by chajchaj##                  stack=False,
#1012 by chajchaj##                  padding_value=0,
#1012 by chajchaj##                  cpu_only=False,
#1012 by chajchaj##                  pad_dims=2):
#1012 by chajchaj##         self._data = data
#1012 by chajchaj##         self._cpu_only = cpu_only
#1012 by chajchaj##         self._stack = stack
#1012 by chajchaj##         self._padding_value = padding_value
#1012 by chajchaj##         assert pad_dims in [None, 1, 2, 3]
#1012 by chajchaj##         self._pad_dims = pad_dims
#1012 by chajchaj## 
#1012 by chajchaj##     def __repr__(self):
#1012 by chajchaj##         return f'{self.__class__.__name__}({repr(self.data)})'
#1012 by chajchaj## 
#1012 by chajchaj##     def __len__(self):
#1012 by chajchaj##         return len(self._data)
#1012 by chajchaj## 
#1012 by chajchaj##     @property
#1012 by chajchaj##     def data(self):
#1012 by chajchaj##         return self._data
#1012 by chajchaj## 
#1012 by chajchaj##     @property
#1012 by chajchaj##     def datatype(self):
#1012 by chajchaj##         if isinstance(self.data, paddle.Tensor):
#1012 by chajchaj##             return self.data.type()
#1012 by chajchaj##         else:
#1012 by chajchaj##             return type(self.data)
#1012 by chajchaj## 
#1012 by chajchaj##     @property
#1012 by chajchaj##     def cpu_only(self):
#1012 by chajchaj##         return self._cpu_only
#1012 by chajchaj## 
#1012 by chajchaj##     @property
#1012 by chajchaj##     def stack(self):
#1012 by chajchaj##         return self._stack
#1012 by chajchaj## 
#1012 by chajchaj##     @property
#1012 by chajchaj##     def padding_value(self):
#1012 by chajchaj##         return self._padding_value
#1012 by chajchaj## 
#1012 by chajchaj##     @property
#1012 by chajchaj##     def pad_dims(self):
#1012 by chajchaj##         return self._pad_dims
#1012 by chajchaj## 
#1012 by chajchaj##     # @assert_tensor_type
#1012 by chajchaj##     # def size(self, *args, **kwargs):
#1012 by chajchaj##     #     return self.data.size(*args, **kwargs)
#1012 by chajchaj##     #
#1012 by chajchaj##     # @assert_tensor_type
#1012 by chajchaj##     # def dim(self):
#1012 by chajchaj##     #     return self.data.dim()
#1012 by chajchaj## 
#1012 by chajchaj## @PIPELINES.register()
#1012 by chajchaj## class ToDataContainer:
#1012 by chajchaj##     """Convert the data to DataContainer.
#1012 by chajchaj## 
#1012 by chajchaj##     Args:
#1012 by chajchaj##         fields (Sequence[dict]): Required fields to be converted
#1012 by chajchaj##             with keys and attributes. E.g.
#1012 by chajchaj##             fields=(dict(key='gt_bbox', stack=False),).
#1012 by chajchaj##             Note that key can also be a list of keys, if so, every tensor in
#1012 by chajchaj##             the list will be converted to DataContainer.
#1012 by chajchaj##     """
#1012 by chajchaj## 
#1012 by chajchaj##     def __init__(self, fields):
#1012 by chajchaj##         self.fields = fields
#1012 by chajchaj## 
#1012 by chajchaj##     def __call__(self, results):
#1012 by chajchaj##         """Performs the ToDataContainer formating.
#1012 by chajchaj## 
#1012 by chajchaj##         Args:
#1012 by chajchaj##             results (dict): The resulting dict to be modified and passed
#1012 by chajchaj##                 to the next transform in pipeline.
#1012 by chajchaj##         """
#1012 by chajchaj##         for field in self.fields:
#1012 by chajchaj##             _field = field.copy()
#1012 by chajchaj##             key = _field.pop('key')
#1012 by chajchaj##             if isinstance(key, list):
#1012 by chajchaj##                 for item in key:
#1012 by chajchaj##                     results[item] = DataContainer(results[item], **_field)
#1012 by chajchaj##             else:
#1012 by chajchaj##                 results[key] = DataContainer(results[key], **_field)
#1012 by chajchaj##         return results
#1012 by chajchaj## 
#1012 by chajchaj##     def __repr__(self):
#1012 by chajchaj##         return self.__class__.__name__ + f'(fields={self.fields})'
#1012 by chajchaj## 
#1012 by chajchaj## @PIPELINES.register()
#1012 by chajchaj## class Collect:
#1012 by chajchaj##     """Collect data from the loader relevant to the specific task.
#1012 by chajchaj## 
#1012 by chajchaj##     This keeps the items in ``keys`` as it is, and collect items in
#1012 by chajchaj##     ``meta_keys`` into a meta item called ``meta_name``.This is usually
#1012 by chajchaj##     the last stage of the data loader pipeline.
#1012 by chajchaj##     For example, when keys='imgs', meta_keys=('filename', 'label',
#1012 by chajchaj##     'original_shape'), meta_name='img_metas', the results will be a dict with
#1012 by chajchaj##     keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
#1012 by chajchaj##     another dict with keys 'filename', 'label', 'original_shape'.
#1012 by chajchaj## 
#1012 by chajchaj##     Args:
#1012 by chajchaj##         keys (Sequence[str]): Required keys to be collected.
#1012 by chajchaj##         meta_name (str): The name of the key that contains meta infomation.
#1012 by chajchaj##             This key is always populated. Default: "img_metas".
#1012 by chajchaj##         meta_keys (Sequence[str]): Keys that are collected under meta_name.
#1012 by chajchaj##             The contents of the ``meta_name`` dictionary depends on
#1012 by chajchaj##             ``meta_keys``.
#1012 by chajchaj##             By default this includes:
#1012 by chajchaj## 
#1012 by chajchaj##             - "filename": path to the image file
#1012 by chajchaj##             - "label": label of the image file
#1012 by chajchaj##             - "original_shape": original shape of the image as a tuple
#1012 by chajchaj##                 (h, w, c)
#1012 by chajchaj##             - "img_shape": shape of the image input to the network as a tuple
#1012 by chajchaj##                 (h, w, c).  Note that images may be zero padded on the
#1012 by chajchaj##                 bottom/right, if the batch tensor is larger than this shape.
#1012 by chajchaj##             - "pad_shape": image shape after padding
#1012 by chajchaj##             - "flip_direction": a str in ("horiziontal", "vertival") to
#1012 by chajchaj##                 indicate if the image is fliped horizontally or vertically.
#1012 by chajchaj##             - "img_norm_cfg": a dict of normalization information:
#1012 by chajchaj##                 - mean - per channel mean subtraction
#1012 by chajchaj##                 - std - per channel std divisor
#1012 by chajchaj##                 - to_rgb - bool indicating if bgr was converted to rgb
#1012 by chajchaj##         nested (bool): If set as True, will apply data[x] = [data[x]] to all
#1012 by chajchaj##             items in data. The arg is added for compatibility. Default: False.
#1012 by chajchaj##     """
#1012 by chajchaj## 
#1012 by chajchaj##     def __init__(self,
#1012 by chajchaj##                  keys,
#1012 by chajchaj##                  meta_keys=('filename', 'label', 'original_shape', 'img_shape',
#1012 by chajchaj##                             'pad_shape', 'flip_direction', 'img_norm_cfg'),
#1012 by chajchaj##                  meta_name='img_metas',
#1012 by chajchaj##                  nested=False):
#1012 by chajchaj##         self.keys = keys
#1012 by chajchaj##         self.meta_keys = meta_keys
#1012 by chajchaj##         self.meta_name = meta_name
#1012 by chajchaj##         self.nested = nested
#1012 by chajchaj## 
#1012 by chajchaj##     def __call__(self, results):
#1012 by chajchaj##         """Performs the Collect formating.
#1012 by chajchaj## 
#1012 by chajchaj##         Args:
#1012 by chajchaj##             results (dict): The resulting dict to be modified and passed
#1012 by chajchaj##                 to the next transform in pipeline.
#1012 by chajchaj##         """
#1012 by chajchaj##         data = {}
#1012 by chajchaj##         for key in self.keys:
#1012 by chajchaj##             data[key] = results[key]
#1012 by chajchaj## 
#1012 by chajchaj##         if len(self.meta_keys) != 0:
#1012 by chajchaj##             meta = {}
#1012 by chajchaj##             for key in self.meta_keys:
#1012 by chajchaj##                 meta[key] = results[key]
#1012 by chajchaj##             data[self.meta_name] = DataContainer(meta, cpu_only=True)
#1012 by chajchaj##         if self.nested:
#1012 by chajchaj##             for k in data:
#1012 by chajchaj##                 data[k] = [data[k]]
#1012 by chajchaj## 
#1012 by chajchaj##         return data
#1012 by chajchaj## 
#1012 by chajchaj##     def __repr__(self):
#1012 by chajchaj##         return (f'{self.__class__.__name__}('
#1012 by chajchaj##                 f'keys={self.keys}, meta_keys={self.meta_keys}, '
#1012 by chajchaj##                 f'nested={self.nested})')
#1012 by chajchaj## 
#1012 by chajchaj## 
