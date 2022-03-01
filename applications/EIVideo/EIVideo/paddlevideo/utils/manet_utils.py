from __future__ import absolute_import

import json
import math
import os
import pickle
import warnings

import numpy
import numpy as np
from numpy import inf
from paddle import Tensor, concat, reshape, nn
import paddle

from typing import Union, Iterable

# from reprod_log.compare import compute_diff
# from reprod_log.utils import check_print_diff, np2torch, np2paddle, torch2np, paddle2np

_tensor_or_tensors = Union[paddle.Tensor, Iterable[paddle.Tensor]]
_palette = [
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128,
    128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0,
    128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0, 128, 64, 0, 0, 191,
    0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 22, 22, 22, 23, 23, 23, 24, 24,
    24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30,
    31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37,
    37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43,
    43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49,
    50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56,
    56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62,
    62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68,
    69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75,
    75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81,
    81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87,
    88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94,
    94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100,
    100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105,
    105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110,
    110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115,
    115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120,
    120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125,
    125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130,
    130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135,
    135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140,
    140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144, 145,
    145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150,
    150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155,
    155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160,
    160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165,
    165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170,
    170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175,
    175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179, 180,
    180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185,
    185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190,
    190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194, 195,
    195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200,
    200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205,
    205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209, 210,
    210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215,
    215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220,
    220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225,
    225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230,
    230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235,
    235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240,
    240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245,
    245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250,
    250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255,
    255, 255
]

# paddle.set_device('gpu') if paddle.is_compiled_with_cuda() else paddle.set_device('cpu')

import paddle
import PIL
import numbers
import numpy as np
from PIL import Image
from paddle.vision.transforms import BaseTransform
from paddle.vision.transforms import functional as F

import numpy as np
from scipy.ndimage import interpolation, binary_dilation
try:
    from skimage import morphology, transform
except ImportError as e:
    print(
        f"{e}, [scikit-image] package and it's dependencies is required for EIVideo."
    )
import paddle
import cv2
import random


####
def mask_damager(labels=None, p_black=0.2):
    scales = (0.8, 1.0, 1.2)
    kernel_size = random.randint(10, 15)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if random.random() < p_black:
        final_label = paddle.zeros_like(labels)
        final_label = final_label.squeeze().numpy()
    else:
        prot = random.randint(5, 15)
        nrot = random.randint(-15, -5)
        rots = [prot, nrot, 0]
        rot = rots[random.randint(0, 2)]

        sc = scales[random.randint(0, 2)]
        _, _, h, w = labels.shape
        tmp = labels.squeeze()

        tmp = tmp.unsqueeze(-1)
        tmp = tmp.numpy().astype(np.uint8)
        morph_p = random.random()
        if morph_p < 0.5:
            tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, kernel)
        else:
            tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel)

        tmp = tmp.astype(np.uint8)
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, rot, sc)
        final_label = cv2.warpAffine(tmp, M, (w, h), cv2.INTER_NEAREST)

    return final_label


color_map = [
    [0, 0, 0],
    [255, 127, 0],
    [30, 144, 255],
    [186, 85, 211],
    [255, 105, 180],
    [192, 255, 62],
    [255, 105, 180],
    [50, 255, 255],
]

color_map_np = np.array(color_map)


def overlay_davis(image, mask, alpha=0.5):
    """ Overlay segmentation on top of RGB image. from davis official"""
    im_overlay = image.copy()
    mask = mask.astype('uint8')
    colored_mask = color_map_np[mask]
    foreground = image * alpha + (1 - alpha) * colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    countours = binary_dilation(binary_mask) ^ binary_mask
    im_overlay[countours, :] = 0
    return im_overlay.astype(image.dtype)


# TODO
def submit_masks(masks, images, inter_file_path):
    overlays = []
    save_result_path = os.path.join(inter_file_path, 'result')
    os.makedirs(save_result_path, exist_ok=True)
    for imgname, (mask, image) in enumerate(zip(masks, images)):
        overlay = overlay_davis(image, mask)
        overlays.append(overlay.tolist())
        overlay = Image.fromarray(overlay)
        imgname = str(imgname)
        while len(imgname) < 5:
            imgname = '0' + imgname
        overlay.save(os.path.join(save_result_path, imgname + '.png'))
    result = {'overlays': overlays}
    # result = {'masks': masks.tolist()}
    with open(os.path.join(save_result_path, 'masks.json'), 'w') as f:
        json.dump(result, f)


def load_video(path, min_side=None):
    frame_list = []
    cap = cv2.VideoCapture(path)
    while (cap.isOpened()):
        _, frame = cap.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if min_side:
            h, w = frame.shape[:2]
            new_w = (w * min_side // min(w, h))
            new_h = (h * min_side // min(w, h))
            frame = cv2.resize(frame, (new_w, new_h),
                               interpolation=cv2.INTER_CUBIC)
            # .transpose([2, 0, 1])
        frame_list.append(frame)
    frames = np.stack(frame_list, axis=0)
    return frames


def get_scribbles():
    for i in range(8):
        with open(f'/home/lc/paddlevideo/data/bike-packing/lable/{i + 1}.json'
                  ) as f:
            scribbles = json.load(f)
            first_scribble = not i
            yield scribbles, first_scribble


def get_images(sequence='bike-packing'):
    img_path = os.path.join('/home/lc/paddlevideo/data', sequence.strip(),
                            'frame')
    img_files = os.listdir(img_path)
    img_files.sort()
    files = []
    for img in img_files:
        img_file = np.array(Image.open(os.path.join(img_path, img)))
        files.append(img_file)
    return np.array(files)


def rough_ROI(ref_scribble_labels):
    #### b*1*h*w
    dist = 20
    b, _, h, w = ref_scribble_labels.shape
    filter_ = paddle.zeros_like(ref_scribble_labels)
    to_fill = paddle.zeros_like(ref_scribble_labels)
    for i in range(b):
        no_background = (ref_scribble_labels[i] != -1)
        no_background = no_background.squeeze(0)

        no_b = no_background.nonzero()
        (h_min, w_min) = paddle.min(no_b, 0)
        (h_max, w_max) = paddle.max(no_b, 0)
        filter_[i, 0,
                max(h_min - dist, 0):min(h_max + dist, h - 1),
                max(w_min - dist, 0):min(w_max + dist, w - 1)] = 1

    final_scribble_labels = paddle.where(byte_(filter_), ref_scribble_labels,
                                         to_fill)
    return final_scribble_labels


import os.path as osp


def load(file_name, model, **cfg):
    if not osp.isfile(file_name):
        raise IOError(f'{file_name} not exist')
    try:
        state_dicts_ = paddle.load(file_name)['state_dict']
    except:
        state_dicts_ = paddle.load(file_name)
    state_dicts = {}
    for k in model.keys():
        if 'num_batches_tracked' not in k:
            if ('head.' + k) not in state_dicts_.keys():
                if k not in state_dicts_.keys():
                    print(f'model -----{k} -------is not in pretrained')
                else:
                    state_dicts[k] = state_dicts_[k]
            else:
                state_dicts[k] = state_dicts_['head.' + k]
    write_dict(state_dicts, 'state_dicts.txt', **cfg)
    write_dict(model, 'model.txt', **cfg)
    return state_dicts


#####
def write_dict(state_dict, file_name, **cfg):
    lines = []
    tot = 0
    for k, v in state_dict.items():
        # 目前只发现了torch和paddle模型参数命名的这三种不一致
        # 不一致1
        if 'num_batches_tracked' in k:
            tot += 1
            continue
        try:
            line = str(k) + '\t' + str(v.cpu().detach().numpy().shape) + '\n'
        except:
            line = str(k) + '\t' + str(v.shape) + '\n'
        lines.append(line)
    # with open(cfg.get("output_dir", f"./output/{file_name}"), 'w') as f:
    #     f.writelines(lines)
    # print('%d num_batches_tracked skipped' % tot)


def damage_masks(labels, shift=True, scale=True, rotate=True):
    """
    Args:
    labels: numpy array (batch_size * 1 * h * w)
    """
    bs, _, h, w = labels.shape
    labels = labels.transpose([0, 2, 3, 1])
    labels = labels.numpy()
    final_label = []
    for i in range(bs):
        label = labels[i]
        damaged_label = damage_masks_np(label, shift, scale, rotate)
        final_label.append(damaged_label)
    final_label = np.array(final_label)
    final_label = paddle.to_tensor(final_label)
    final_label = final_label.transpose([0, 3, 1, 2])
    return final_label


def damage_masks_np(labels, shift=True, scale=True, rotate=True):
    """Performs the actual mask damaging in numpy.
    Args:
    labels: Int32 numpy array of shape (height, width, 1).
    shift: Boolean, whether to damage the masks by shifting.
    scale: Boolean, whether to damage the masks by scaling.
    rotate: Boolean, whether to damage the masks by rotation.
    dilate: Boolean, whether to damage the masks by dilation.
    Returns:
    The damaged version of labels.
    """
    unique_labels = np.unique(labels)
    unique_labels = np.setdiff1d(unique_labels, [0])
    # Shuffle to get random depth ordering when combining together.
    np.random.shuffle(unique_labels)
    damaged_labels = np.zeros_like(labels)
    for l in unique_labels:
        obj_mask = (labels == l)
        damaged_obj_mask = _damage_single_object_mask(obj_mask, shift, scale,
                                                      rotate)
        damaged_labels[damaged_obj_mask] = l
    return damaged_labels


def _damage_single_object_mask(mask, shift, scale, rotate):
    """Performs mask damaging in numpy for a single object.
    Args:
    mask: Boolean numpy array of shape(height, width, 1).
    shift: Boolean, whether to damage the masks by shifting.
    scale: Boolean, whether to damage the masks by scaling.
    rotate: Boolean, whether to damage the masks by rotation.
    dilate: Boolean, whether to damage the masks by dilation.
    Returns:
    The damaged version of mask.
    """
    if shift:
        mask = _shift_mask(mask)
    if scale:
        mask = _scale_mask(mask)
    if rotate:
        mask = _rotate_mask(mask)
    return mask


def _shift_mask(mask, max_shift_factor=0.05):
    """Damages a mask for a single object by randomly shifting it in numpy.
    Args:
    mask: Boolean numpy array of shape(height, width, 1).
    max_shift_factor: Float scalar, the maximum factor for random shifting.
    Returns:
    The shifted version of mask.
    """
    nzy, nzx, _ = mask.nonzero()
    h = nzy.max() - nzy.min()
    w = nzx.max() - nzx.min()
    size = np.sqrt(h * w)
    offset = np.random.uniform(-size * max_shift_factor,
                               size * max_shift_factor, 2)
    shifted_mask = interpolation.shift(np.squeeze(mask, axis=2),
                                       offset,
                                       order=0).astype('bool')[..., np.newaxis]
    return shifted_mask


def _scale_mask(mask, scale_amount=0.025):
    """Damages a mask for a single object by randomly scaling it in numpy.
    Args:
    mask: Boolean numpy array of shape(height, width, 1).
    scale_amount: Float scalar, the maximum factor for random scaling.
    Returns:
    The scaled version of mask.
    """
    nzy, nzx, _ = mask.nonzero()
    cy = 0.5 * (nzy.max() - nzy.min())
    cx = 0.5 * (nzx.max() - nzx.min())
    scale_factor = np.random.uniform(1.0 - scale_amount, 1.0 + scale_amount)
    shift = transform.SimilarityTransform(translation=[-cx, -cy])
    inv_shift = transform.SimilarityTransform(translation=[cx, cy])
    s = transform.SimilarityTransform(scale=[scale_factor, scale_factor])
    m = (shift + (s + inv_shift)).inverse
    scaled_mask = transform.warp(mask, m) > 0.5
    return scaled_mask


def _rotate_mask(mask, max_rot_degrees=3.0):
    """Damages a mask for a single object by randomly rotating it in numpy.
    Args:
    mask: Boolean numpy array of shape(height, width, 1).
    max_rot_degrees: Float scalar, the maximum number of degrees to rotate.
    Returns:
    The scaled version of mask.
    """
    cy = 0.5 * mask.shape[0]
    cx = 0.5 * mask.shape[1]
    rot_degrees = np.random.uniform(-max_rot_degrees, max_rot_degrees)
    shift = transform.SimilarityTransform(translation=[-cx, -cy])
    inv_shift = transform.SimilarityTransform(translation=[cx, cy])
    r = transform.SimilarityTransform(rotation=np.deg2rad(rot_degrees))
    m = (shift + (r + inv_shift)).inverse
    scaled_mask = transform.warp(mask, m) > 0.5
    return scaled_mask


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import numpy as np


def label2colormap(label):
    m = label.astype(np.uint8)
    r, c = m.shape
    cmap = np.zeros((r, c, 3), dtype=np.uint8)
    cmap[:, :, 0] = (m & 1) << 7 | (m & 8) << 3 | (m & 64) >> 1
    cmap[:, :, 1] = (m & 2) << 6 | (m & 16) << 2 | (m & 128) >> 2
    cmap[:, :, 2] = (m & 4) << 5 | (m & 32) << 1
    return cmap


def torch2paddle(data):
    try:
        import torch
        if isinstance(data, dict):
            np_data = {}
            for k, v in data.items():
                np_data[k] = paddle.to_tensor(v.detach().numpy())
            return np_data
        else:
            return paddle.to_tensor(data.detach().numpy())
    except:
        pass


def fill_(tensor: Tensor, value):
    return tensor.set_value(paddle.full_like(tensor, value))


def zero_(tensor: Tensor):
    return tensor.set_value(paddle.zeros_like(tensor))


def float_(tensor: Tensor):
    return paddle.to_tensor(tensor, dtype='float32')


def long_(tensor: Tensor):
    return paddle.to_tensor(tensor, dtype='int64')


def int_(tensor: Tensor):
    return paddle.to_tensor(tensor, dtype='int32')


def byte_(tensor: Tensor):
    return paddle.to_tensor(tensor, dtype='bool')


class ToPILImage(BaseTransform):
    def __init__(self, mode=None, keys=None):
        super(ToPILImage, self).__init__(keys)

    def _apply_image(self, pic):
        """
        Args:
            pic (Tensor|np.ndarray): Image to be converted to PIL Image.
        Returns:
            PIL: Converted image.
        """
        if not (isinstance(pic, paddle.Tensor) or isinstance(pic, np.ndarray)):
            raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(
                type(pic)))

        elif isinstance(pic, paddle.Tensor):
            if pic.ndimension() not in {2, 3}:
                raise ValueError(
                    'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                        pic.ndimension()))

            elif pic.ndimension() == 2:
                # if 2D image, add channel dimension (CHW)
                pic = pic.unsqueeze(0)

        elif isinstance(pic, np.ndarray):
            if pic.ndim not in {2, 3}:
                raise ValueError(
                    'pic should be 2/3 dimensional. Got {} dimensions.'.format(
                        pic.ndim))

            elif pic.ndim == 2:
                # if 2D image, add channel dimension (HWC)
                pic = np.expand_dims(pic, 2)

        npimg = pic
        if isinstance(pic, paddle.Tensor) and "float" in str(
                pic.numpy().dtype) and self.mode != 'F':
            pic = pic.mul(255).byte()
        if isinstance(pic, paddle.Tensor):
            npimg = np.transpose(pic.numpy(), (1, 2, 0))

        if not isinstance(npimg, np.ndarray):
            raise TypeError(
                'Input pic must be a paddle.Tensor or NumPy ndarray, ' +
                'not {}'.format(type(npimg)))

        if npimg.shape[2] == 1:
            expected_mode = None
            npimg = npimg[:, :, 0]
            if npimg.dtype == np.uint8:
                expected_mode = 'L'
            elif npimg.dtype == np.int16:
                expected_mode = 'I;16'
            elif npimg.dtype == np.int32:
                expected_mode = 'I'
            elif npimg.dtype == np.float32:
                expected_mode = 'F'
            if self.mode is not None and self.mode != expected_mode:
                raise ValueError(
                    "Incorrect self.mode ({}) supplied for input type {}. Should be {}"
                    .format(self.mode, np.dtype, expected_mode))
            self.mode = expected_mode

        elif npimg.shape[2] == 2:
            permitted_2_channel_modes = ['LA']
            if self.mode is not None and self.mode not in permitted_2_channel_modes:
                raise ValueError(
                    "Only self.modes {} are supported for 2D inputs".format(
                        permitted_2_channel_modes))

            if self.mode is None and npimg.dtype == np.uint8:
                self.mode = 'LA'

        elif npimg.shape[2] == 4:
            permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
            if self.mode is not None and self.mode not in permitted_4_channel_modes:
                raise ValueError(
                    "Only self.modes {} are supported for 4D inputs".format(
                        permitted_4_channel_modes))

            if self.mode is None and npimg.dtype == np.uint8:
                self.mode = 'RGBA'
        else:
            permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
            if self.mode is not None and self.mode not in permitted_3_channel_modes:
                raise ValueError(
                    "Only self.modes {} are supported for 3D inputs".format(
                        permitted_3_channel_modes))
            if self.mode is None and npimg.dtype == np.uint8:
                self.mode = 'RGB'

        if self.mode is None:
            raise TypeError('Input type {} is not supported'.format(
                npimg.dtype))

        return Image.fromarray(npimg, mode=self.mode)


class Identity(nn.Layer):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)
    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def convert(data: dict, to, dtype=None):
    assert isinstance(data, dict)
    input = {}
    for k, v in data.items():

        if 'paddle' == to:
            if isinstance(v, np.ndarray):
                if dtype is not None:
                    input[k] = paddle.to_tensor(v.astype(dtype))
                else:
                    input[k] = paddle.to_tensor(v)
            else:
                input[k] = v
        elif 'torch' == to:
            try:
                import torch
                if isinstance(v, np.ndarray):
                    if dtype is not None:
                        input[k] = torch.tensor(v.astype(dtype))
                    else:
                        input[k] = torch.tensor(v)
                else:
                    input[k] = v
            except:
                pass
        else:
            if isinstance(v, np.ndarray):
                input[k] = v.astype(to)
            else:
                input[k] = v
    return input


def clip_grad_norm_(parameters: _tensor_or_tensors,
                    max_norm: float,
                    norm_type: float = 2.0,
                    error_if_nonfinite: bool = False) -> paddle.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:``parameters`` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    import time
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    detached_grads = [p.grad.detach() for p in parameters]

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return paddle.to_tensor(0.)
    # device = paddle.get_device()  # parameters[0].grad.device
    if norm_type == inf:
        norms = [p.abs().max() for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else paddle.max(
            paddle.stack(norms))
    else:
        #         tik = time.time()
        total_norm = paddle.norm(
            paddle.stack([paddle.norm(g, norm_type) for g in detached_grads]),
            norm_type)
    #         total_norm = paddle.norm(paddle.stack([paddle.sqrt(paddle.sum(g*g)) for g in detached_grads]), norm_type)  # fixed.
    #         print(time.time() - tik)
    if error_if_nonfinite and paddle.logical_or(total_norm.isnan(),
                                                total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = paddle.clip(clip_coef, max=1.0)
    for i, p in enumerate(parameters):
        #         p.set_value(paddle.multiply(p, clip_coef_clamped))
        p.grad.set_value(detached_grads[i] * clip_coef_clamped)  # fixed
    #         p.grad.detach().mul_(clip_coef_clamped
    return total_norm


# def max(a: paddle.Tensor, axis=0, keepdim=True):
#     """ndarray=numpy.array([[1, 2, 3, 4],
#            [4, 3, 2, 1],
#            [5, 6, 7, 8],
#            [8, 7, 6, 5]])
#     np.where(ndarray == np.max(ndarray))
#     (array([2, 3]), array([3, 0]))
#     ndarray[np.where(ndarray == np.max(ndarray))]
#     array([8, 8])
#     """
#     max_ = a.max(axis).unsqueeze(-1)
#     index = paddle.argmax(a, axis=axis, keepdim=keepdim)
#     max_ = max_.numpy()
#     index = index.numpy()
#     # index = paddle.argmax(a, axis=axis, keepdim=keepdim)[-1].flatten()
#     return max_, index


def gather(tmp: paddle.Tensor, ind: paddle.Tensor):
    shape = tmp.shape
    tmp = paddle.to_tensor(tmp)
    ind = paddle.to_tensor(ind)
    if len(shape) == 2:
        b = shape[0]
        return concat([
            reshape(paddle.gather(tmp[i, :], ind[i, :]), [1, -1])
            for i in range(b)
        ],
                      axis=0)
    elif len(shape) == 3:
        out = []
        for i in range(tmp.shape[0]):
            _ = paddle.index_sample(tmp[i], ind[i])
            out.append(_)
        return paddle.to_tensor(out)
    elif len(shape) == 4:
        b, c, d = shape[:3]
        return concat([
            reshape(
                concat([
                    reshape(
                        concat([
                            reshape(
                                paddle.gather(tmp[i, j, k, :], ind[i, j, k, :]),
                                [1, -1]) for k in range(d)
                        ],
                               axis=0), [1, d, -1]) for j in range(c)
                ],
                       axis=0), [1, c, d, -1]) for i in range(b)
        ],
                      axis=0)
    else:
        pass


# These no_grad_* functions are necessary as wrappers around the parts of these
# functions that use `with torch.no_grad()`. The JIT doesn't support context
# managers, so these need to be implemented as builtins. Using these wrappers
# lets us keep those builtins small and re-usable.
def _no_grad_uniform_(tensor, a, b):
    with paddle.no_grad():
        tensor.set_value(paddle.uniform(tensor.shape, min=a, max=b))
        return tensor


def _no_grad_normal_(tensor, mean, std):
    with paddle.no_grad():
        tensor.set_value(paddle.normal(shape=tensor.shape, mean=mean, std=std))
        return tensor


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    from scipy import special

    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2)

    with paddle.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.set_value(
            paddle.uniform(tensor.shape, min=2 * l - 1, max=2 * u - 1))
        # tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        # tensor.erfinv_()  # paddle 无
        tensor.set_value(special.erfinv(tensor))

        # Transform to proper mean, std
        # tensor.mul_(std * math.sqrt(2.))
        tensor.set_value(tensor.multiply(paddle.to_tensor(std * math.sqrt(2.))))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clip_(min=a, max=b)
        return tensor


def _no_grad_fill_(tensor, val):
    with paddle.no_grad():
        tensor.set_value(paddle.full_like(tensor, fill_value=val))
        return tensor


def _no_grad_zero_(tensor):
    with paddle.no_grad():
        tensor.set_value(paddle.zeros_like(tensor))
        return tensor


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    """
    linear_fns = [
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d'
    ]
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(
                param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(
                "negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def uniform_(tensor: Tensor, a: float = 0., b: float = 1.) -> Tensor:
    r"""Fills the input Tensor with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.uniform_(w)
    """
    return _no_grad_uniform_(tensor, a, b)


def normal_(tensor: Tensor, mean: float = 0., std: float = 1.) -> Tensor:
    r"""Fills the input Tensor with values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.normal_(w)
    """
    return _no_grad_normal_(tensor, mean, std)


def trunc_normal_(tensor: Tensor,
                  mean: float = 0.,
                  std: float = 1.,
                  a: float = -2.,
                  b: float = 2.) -> Tensor:
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def constant_(tensor: Tensor, val: float) -> Tensor:
    r"""Fills the input Tensor with the value :math:`\text{val}`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.constant_(w, 0.3)
    """
    return _no_grad_fill_(tensor, val)


def ones_(tensor: Tensor) -> Tensor:
    r"""Fills the input Tensor with the scalar value `1`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.ones_(w)
    """
    return _no_grad_fill_(tensor, 1.)


def zeros_(tensor: Tensor) -> Tensor:
    r"""Fills the input Tensor with the scalar value `0`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.zeros_(w)
    """
    return _no_grad_zero_(tensor)


def eye_(tensor):
    r"""Fills the 2-dimensional input `Tensor` with the identity
    matrix. Preserves the identity of the inputs in `Linear` layers, where as
    many inputs are preserved as possible.

    Args:
        tensor: a 2-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.eye_(w)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    with paddle.no_grad():
        tensor.set_value(paddle.eye(*tensor.shape))
    return tensor


def dirac_(tensor, groups=1):
    r"""Fills the {3, 4, 5}-dimensional input `Tensor` with the Dirac
    delta function. Preserves the identity of the inputs in `Convolutional`
    layers, where as many input channels are preserved as possible. In case
    of groups>1, each group of channels preserves identity

    Args:
        tensor: a {3, 4, 5}-dimensional `torch.Tensor`
        groups (optional): number of groups in the conv layer (default: 1)
    Examples:
        >>> w = torch.empty(3, 16, 5, 5)
        >>> nn.init.dirac_(w)
        >>> w = torch.empty(3, 24, 5, 5)
        >>> nn.init.dirac_(w, 3)
    """
    dimensions = tensor.ndimension()
    if dimensions not in [3, 4, 5]:
        raise ValueError(
            "Only tensors with 3, 4, or 5 dimensions are supported")

    sizes = tensor.shape

    if sizes[0] % groups != 0:
        raise ValueError('dim 0 must be divisible by groups')

    out_chans_per_grp = sizes[0] // groups
    min_dim = min(out_chans_per_grp, sizes[1])

    with paddle.no_grad():
        tensor.zero_()

        for g in range(groups):
            for d in range(min_dim):
                if dimensions == 3:  # Temporal convolution
                    tensor[g * out_chans_per_grp + d, d,
                           tensor.shape[2] // 2] = 1
                elif dimensions == 4:  # Spatial convolution
                    tensor[g * out_chans_per_grp + d, d, tensor.shape[2] // 2,
                           tensor.shape[3] // 2] = 1
                else:  # Volumetric convolution
                    tensor[g * out_chans_per_grp + d, d, tensor.shape[2] // 2,
                           tensor.shape[3] // 2, tensor.shape[4] // 2] = 1
    return tensor


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.shape[1]  # .size(1)
    num_output_fmaps = tensor.shape[0]  # .size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s  # fixed
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def LongTensor(x):
    return paddle.to_tensor(x, dtype='int64')


def IntTensor(x):
    return paddle.to_tensor(x, dtype='int32')


def xavier_uniform_(tensor: Tensor, gain: float = 1.) -> Tensor:
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)


def xavier_normal_(tensor: Tensor, gain: float = 1.) -> Tensor:
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _no_grad_normal_(tensor, 0., std)


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(
            mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(
        3.0) * std  # Calculate uniform bounds from standard deviation
    with paddle.no_grad():
        tensor.set_value(paddle.uniform(tensor.shape, min=-bound, max=bound))
        return tensor


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with paddle.no_grad():
        tensor.set_value(paddle.normal(shape=tensor.shape, mean=0, std=std))
        return tensor


def orthogonal_(tensor, gain=1):
    r"""Fills the input `Tensor` with a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.orthogonal_(w)
    """
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.shape[0]  # .size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = paddle.to_tensor(np.linalg.qr(flattened.numpy()))
    # q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = paddle.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()

    with paddle.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def sparse_(tensor, sparsity, std=0.01):
    r"""Fills the 2D input `Tensor` as a sparse matrix, where the
    non-zero elements will be drawn from the normal distribution
    :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))

    with paddle.no_grad():
        tensor.normal_(0, std)
        for col_idx in range(cols):
            row_indices = paddle.randperm(rows)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor


# for backward compatibility
def _make_deprecate(meth):
    new_name = meth.__name__
    old_name = new_name[:-1]

    def deprecated_init(*args, **kwargs):
        warnings.warn(
            "nn.init.{} is now deprecated in favor of nn.init.{}.".format(
                old_name, new_name),
            stacklevel=2)
        return meth(*args, **kwargs)

    deprecated_init.__doc__ = r"""
    {old_name}(...)

    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.{new_name}`.

    See :func:`~torch.nn.init.{new_name}` for details.""".format(
        old_name=old_name, new_name=new_name)
    deprecated_init.__name__ = old_name
    return deprecated_init


# uniform = _make_deprecate(uniform_)
# normal = _make_deprecate(normal_)
# constant = _make_deprecate(constant_)
# eye = _make_deprecate(eye_)
# dirac = _make_deprecate(dirac_)
# xavier_uniform = _make_deprecate(xavier_uniform_)
# xavier_normal = _make_deprecate(xavier_normal_)
# kaiming_uniform = _make_deprecate(kaiming_uniform_)
# kaiming_normal = _make_deprecate(kaiming_normal_)
# orthogonal = _make_deprecate(orthogonal_)
# sparse = _make_deprecate(sparse_)
