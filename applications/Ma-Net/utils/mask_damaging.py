import numpy as np
from scipy.ndimage import interpolation
try:
    from skimage import morphology, transform
except ImportError as e:
    print(
        f"{e}, [scikit-image] package and it's dependencies is required for MA-Net."
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


#####


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
