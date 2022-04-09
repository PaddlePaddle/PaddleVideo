#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import paddle
from scipy.signal import convolve2d
from paddlevideo.utils import get_logger


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    im1 = np.clip(im1, 0, 1)
    im2 = np.clip(im2, 0, 1)

    im1 = np.around(im1, decimals=4)
    im2 = np.around(im2, decimals=4)
    M, N = im1.shape
    C1 = (k1 * L)**2
    C2 = (k2 * L)**2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'same')
    mu2 = filter2(im2, window, 'same')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter2(im1 * im1, window, 'same') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'same') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'same') - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)


def psnr(pred, gt):
    #pred=pred.clamp(0,1).cpu().numpy()
    pred = paddle.clip(pred, min=0, max=1)
    gt = paddle.clip(gt, min=0, max=1)
    imdff = np.asarray(pred - gt)
    rmse = math.sqrt(np.mean(imdff**2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)
