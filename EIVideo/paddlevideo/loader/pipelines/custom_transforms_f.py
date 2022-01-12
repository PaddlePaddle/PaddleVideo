import os
import random
import cv2
import numpy as np
import paddle
from PIL import Image
from davisinteractive.utils.operations import bresenham

from ..registry import PIPELINES

cv2.setNumThreads(0)
NEW_BRANCH = True


@PIPELINES.register()
class RandomScale_manet(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, scales=[0.75, 1, 1.25]):
        self.scales = scales

    def __call__(self, sample):

        # Fixed range of scales
        sc = self.scales[random.randint(0, len(self.scales) - 1)]

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            tmp = sample[elem]

            if elem == 'img1' or elem == 'img2' or elem == 'ref_img':
                flagval = cv2.INTER_CUBIC
            else:
                flagval = cv2.INTER_NEAREST

            tmp = cv2.resize(tmp, None, fx=sc, fy=sc, interpolation=flagval)

            sample[elem] = tmp

        return sample


@PIPELINES.register()
class Resize_manet(object):
    """Rescale the image in a results to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    #        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST
    #        self.fix = fix

    def __call__(self, results):
        img1 = results['img1']
        h, w = img1.shape[:2]
        if self.output_size == (h, w):
            return results

        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        for elem in results.keys():
            if 'meta' in elem:
                continue
            tmp = results[elem]
            if elem == 'img1' or elem == 'img2' or elem == 'ref_img':
                flagval = cv2.INTER_CUBIC
            else:
                flagval = cv2.INTER_NEAREST

            tmp = cv2.resize(tmp, dsize=(new_w, new_h), interpolation=flagval)
            results[elem] = tmp
        return results


@PIPELINES.register()
class RandomCrop_manet(object):
    """Crop randomly the image in a results.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size, step=None):
        assert isinstance(output_size, (int, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.step = step

    def __call__(self, results):

        image = results['img1']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w
        is_contain_obj = False

        #        while (not is_contain_obj) and (step < 5):
        if self.step is None:
            while not is_contain_obj:
                #                step += 1
                top = np.random.randint(0, h - new_h + 1)
                left = np.random.randint(0, w - new_w + 1)
                ref_scribble_label = results['ref_scribble_label']
                new_ref_scribble_label = ref_scribble_label[top:top + new_h,
                                                            left:left + new_w]
                if len(np.unique(new_ref_scribble_label)) == 1:
                    continue
                else:

                    for elem in results.keys():
                        if 'meta' in elem:
                            continue

                        tmp = results[elem]
                        tmp = tmp[top:top + new_h, left:left + new_w]
                        results[elem] = tmp
                    break
        else:
            st = 0
            while not is_contain_obj and st < self.step:
                st += 1
                top = np.random.randint(0, h - new_h + 1)
                left = np.random.randint(0, w - new_w + 1)
                ref_scribble_label = results['ref_scribble_label']
                new_ref_scribble_label = ref_scribble_label[top:top + new_h,
                                                            left:left + new_w]
                if len(np.unique(
                        new_ref_scribble_label)) == 1 or st < self.step - 1:
                    continue
                else:

                    for elem in results.keys():
                        if 'meta' in elem:
                            continue

                        tmp = results[elem]
                        tmp = tmp[top:top + new_h, left:left + new_w]
                        results[elem] = tmp
                    break

        return results


@PIPELINES.register()
class RandomHorizontalFlip_manet(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
    def __init__(self, prob):
        self.p = prob

    def __call__(self, results):

        if random.random() < self.p:
            for elem in results.keys():
                if 'meta' in elem:
                    continue
                tmp = results[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                results[elem] = tmp

        return results


@PIPELINES.register()
class ToTensor_manet(object):
    """Convert ndarrays in results to Tensors."""
    def __call__(self, results):

        for elem in results.keys():
            if 'meta' in elem:
                continue
            tmp = results[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]
            else:
                tmp = tmp / 255.
                tmp -= (0.485, 0.456, 0.406)
                tmp /= (0.229, 0.224, 0.225)
            tmp = tmp.transpose([2, 0, 1])
            results[elem] = paddle.to_tensor(tmp)
        return results


def gt_from_scribble(scr, dilation=11, nocare_area=21):
    # Compute foreground
    if scr.max() == 1:
        kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (dilation, dilation))
        fg = cv2.dilate(scr.astype(np.uint8),
                        kernel=kernel_fg).astype(scr.dtype)
    else:
        fg = scr

    # Compute nocare area
    if nocare_area is None:
        nocare = None
    else:
        kernel_nc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (nocare_area, nocare_area))
        nocare = cv2.dilate(fg, kernel=kernel_nc) - fg

    return fg, nocare
