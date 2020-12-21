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
from ..registry import PIPELINES


@PIPELINES.register()
class Mixup(object):
    """
    Mixup operator.
    Args:
        alpha(float): alpha value.
    """
    def __init__(self, alpha=0.2):
        assert alpha > 0., \
                'parameter alpha[%f] should > 0.0' % (alpha)
        self.alpha = alpha

    def __call__(self, batch):
        imgs, labels = list(zip(*batch))
        imgs = np.array(imgs)
        labels = np.array(labels)
        bs = len(batch)
        idx = np.random.permutation(bs)
        lam = np.random.beta(self.alpha, self.alpha)
        lams = np.array([lam] * bs, dtype=np.float32)
        imgs = lam * imgs + (1 - lam) * imgs[idx]
        return list(zip(imgs, labels, labels[idx], lams))


@PIPELINES.register()
class Cutmix(object):
    """ Cutmix operator
    Args:
        alpha(float): alpha value.
    """
    def __init__(self, alpha=0.2):
        assert alpha > 0., \
                'parameter alpha[%f] should > 0.0' % (alpha)
        self.alpha = alpha

    def rand_bbox(self, size, lam):
        """ rand_bbox """
        w = size[2]
        h = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch):
        imgs, labels = list(zip(*batch))
        imgs = np.array(imgs)
        labels = np.array(labels)

        bs = len(batch)
        idx = np.random.permutation(bs)
        lam = np.random.beta(self.alpha, self.alpha)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.shape, lam)
        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[idx, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - (float(bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.shape[-2] * imgs.shape[-1]))
        lams = np.array([lam] * bs, dtype=np.float32)

        return list(zip(imgs, labels, labels[idx], lams))
