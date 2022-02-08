import os
import random
import cv2
import numpy as np
import paddle
from PIL import Image
import dataloaders.helpers as helpers
from davisinteractive.utils.operations import bresenham
from paddle.vision.transforms import functional as F

cv2.setNumThreads(0)
NEW_BRANCH = True


class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    #        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST
    #        self.fix = fix

    def __call__(self, sample):
        img1 = sample['img1']
        # img2 = sample['img2']
        # ref_img=sample['ref_img']
        h, w = img1.shape[:2]
        if self.output_size == (h, w):
            return sample

        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        for elem in sample.keys():
            if 'meta' in elem:
                continue
            tmp = sample[elem]
            if elem == 'img1' or elem == 'img2' or elem == 'ref_img':
                flagval = cv2.INTER_CUBIC
            else:
                flagval = cv2.INTER_NEAREST

            tmp = cv2.resize(tmp, dsize=(new_w, new_h), interpolation=flagval)
            sample[elem] = tmp

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size, step=None):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.step = step

    def __call__(self, sample):

        image = sample['img1']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w
        is_contain_obj = False

        if self.step is None:
            while not is_contain_obj:
                #                step += 1
                top = np.random.randint(0, h - new_h + 1)
                left = np.random.randint(0, w - new_w + 1)
                ref_scribble_label = sample['ref_scribble_label']
                new_ref_scribble_label = ref_scribble_label[top:top + new_h,
                                                            left:left + new_w]
                if len(np.unique(new_ref_scribble_label)) == 1:
                    continue
                else:

                    for elem in sample.keys():
                        if 'meta' in elem:
                            continue

                        tmp = sample[elem]
                        tmp = tmp[top:top + new_h, left:left + new_w]
                        sample[elem] = tmp
                    break
        else:
            st = 0
            while not is_contain_obj and st < self.step:
                st += 1
                top = np.random.randint(0, h - new_h + 1)
                left = np.random.randint(0, w - new_w + 1)
                ref_scribble_label = sample['ref_scribble_label']
                new_ref_scribble_label = ref_scribble_label[top:top + new_h,
                                                            left:left + new_w]
                if len(np.unique(
                        new_ref_scribble_label)) == 1 or st < self.step - 1:
                    continue
                else:

                    for elem in sample.keys():
                        if 'meta' in elem:
                            continue

                        tmp = sample[elem]
                        tmp = tmp[top:top + new_h, left:left + new_w]
                        sample[elem] = tmp
                    break

        return sample


class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0]) / 2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'meta' in elem:
                continue

            tmp = sample[elem]

            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert (center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp

        return sample


class RandomScale(object):
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


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
    def __init__(self, prob):
        self.p = prob

    def __call__(self, sample):

        if random.random() < self.p:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

        return sample


class SubtractMeanImage(object):
    def __init__(self, mean, change_channels=False):
        self.mean = mean
        self.change_channels = change_channels

    def __call__(self, sample):
        for elem in sample.keys():
            if 'image' in elem:
                if self.change_channels:
                    sample[elem] = sample[elem][:, :, [2, 1, 0]]
                sample[elem] = np.subtract(
                    sample[elem], np.array(self.mean, dtype=np.float32))
        return sample

    def __str__(self):
        return 'SubtractMeanImage' + str(self.mean)


class CustomScribbleInteractive(object):
    def __init__(self,
                 scribbles,
                 first_frame,
                 dilation=9,
                 nocare_area=None,
                 bresenham=True,
                 use_previous_mask=False,
                 previous_mask_path=None):

        self.scribbles = scribbles
        self.dilation = dilation
        self.nocare_area = nocare_area
        self.bresenham = bresenham
        self.first_frame = first_frame
        self.use_previous_mask = use_previous_mask
        self.previous_mask_path = previous_mask_path

    def __call__(self, sample):
        meta = sample['meta']
        frame_num = int(meta['frame_id'])

        im_size = meta['im_size']

        # Initialize gt to zeros, no-care areas to ones
        scr_gt = np.zeros(im_size)
        scr_nocare = np.ones(im_size)
        mask = np.zeros(im_size)
        mask_neg = np.zeros(im_size)

        # Get all the scribbles for the current frame
        for scribble in self.scribbles[frame_num]:
            points_scribble = np.round(
                np.array(scribble['path']) * np.array(
                    (im_size[1], im_size[0]))).astype(int)
            if self.bresenham and len(points_scribble) > 1:
                all_points = bresenham(points_scribble)
            else:
                all_points = points_scribble

            # Check if scribble is of same id to mark as foreground, otherwise as background
            if scribble['object_id'] == meta['obj_id']:
                mask[all_points[:, 1] - 1, all_points[:, 0] - 1] = 1
            else:
                mask_neg[all_points[:, 1] - 1, all_points[:, 0] - 1] = 1
        if self.nocare_area is None:
            nz = np.where(mask > 0)
            nocare_area = int(.5 * np.sqrt(
                (nz[0].max() - nz[0].min()) * (nz[1].max() - nz[1].min())))
        else:
            nocare_area = 100

        # In case we are reading the first human annotation round
        if frame_num == self.first_frame:
            # Compute dilated foreground, background, and no-care area
            scr_gt, scr_nocare = helpers.gt_from_scribble(
                mask, dilation=self.dilation, nocare_area=nocare_area)
            scr_gt_neg, _ = helpers.gt_from_scribble(mask_neg,
                                                     dilation=self.dilation,
                                                     nocare_area=None)

            # Negative examples included in the training
            scr_gt[scr_gt_neg > 0] = 0
            scr_nocare[scr_gt_neg > 0] = 0

        # For annotation rounds generated by the robot
        else:
            # Compute dilated foreground, background, and no-care area
            scr_gt_extra, _ = helpers.gt_from_scribble(mask,
                                                       dilation=self.dilation,
                                                       nocare_area=None)
            scr_gt_neg, _ = helpers.gt_from_scribble(mask_neg,
                                                     dilation=self.dilation,
                                                     nocare_area=None)

            # Ignore pixels that are not foreground
            if not self.use_previous_mask:
                scr_nocare_extra = 1. - scr_gt_extra
            else:
                scr_nocare_extra = \
                    (cv2.imread(os.path.join(self.previous_mask_path, meta['seq_name'], str(meta['obj_id']),
                                             meta['frame_id'] + '.png'), 0) > 0.8 * 255).astype(np.float32)

            # Negative examples included in training
            scr_gt_extra[scr_gt_neg > 0] = 0
            scr_nocare_extra[scr_gt_neg > 0] = 0

            scr_gt = np.maximum(scr_gt, scr_gt_extra)
            scr_nocare_extra[scr_gt > 0] = 0
            scr_nocare = np.minimum(scr_nocare, scr_nocare_extra)

        sample['scribble_gt'] = scr_gt
        sample['scribble_void_pixels'] = scr_nocare

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]
            else:
                tmp = tmp / 255.
                tmp -= (0.485, 0.456, 0.406)
                tmp /= (0.229, 0.224, 0.225)

            # swap color axis because
            # numpy image: H x W x C
            # paddle image: C X H X W

            tmp = tmp.transpose([2, 0, 1])
            sample[elem] = paddle.to_tensor(tmp)
        return sample


class GenerateEdge(object):
    """
    """
    def __init__(self, edgesize=1):
        self.edgesize = edgesize

    def __call__(self, sample):
        """
        """
        if "label2" in sample:
            label2 = sample['label2']
            kernel_size = 2 * self.edgesize + 1
            maskedge = np.zeros_like(label2)

            maskedge[np.where(label2[:, 1:] != label2[:, :-1])] = 1
            maskedge[np.where(label2[1:, :] != label2[:-1, :])] = 1
            maskedge = cv2.dilate(
                maskedge, np.ones((kernel_size, kernel_size), dtype=np.uint8))
            sample["edge_mask"] = maskedge
        else:
            raise RuntimeError(
                "We need parsing mask to generate the edge mask.")
        return sample


class GenerateEdge_2(object):
    """
    """
    def __init__(self, edgesize=1):
        self.edgesize = edgesize

    def __call__(self, sample):
        """
        """
        if "ref_frame_gt" in sample:
            label2 = sample['ref_frame_gt']
            kernel_size = 2 * self.edgesize + 1
            maskedge = np.zeros_like(label2)

            maskedge[np.where(label2[:, 1:] != label2[:, :-1])] = 1
            maskedge[np.where(label2[1:, :] != label2[:-1, :])] = 1
            maskedge = cv2.dilate(
                maskedge, np.ones((kernel_size, kernel_size), dtype=np.uint8))
            sample["edge_mask"] = maskedge
        else:
            raise RuntimeError(
                "We need parsing mask to generate the edge mask.")
        return sample
