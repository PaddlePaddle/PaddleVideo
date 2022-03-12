# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import paddle

from ..builder import build_loss
from ..registry import HEADS


@HEADS.register()
class WAFPHead(paddle.nn.Layer):
    """WAFP-Net Head.

    Args:
        loss_cfg (Dict[str, str], optional): main loss item. Defaults to dict(name='MSELoss').
        regular_cfg (Dict[str, str], optional): regular loss item. Defaults to dict(name='TVLoss').
    """
    def __init__(self,
                 loss_cfg: Dict[str, str] = dict(name='MSELoss'),
                 regular_cfg: Dict[str, str] = dict(name='TVLoss'),
                 **kwargs):

        super(WAFPHead, self).__init__()
        self.loss_func = build_loss(loss_cfg)
        self.regular_func = build_loss(regular_cfg)

    def forward(self):
        pass

    def loss(self,
             out1: paddle.Tensor,
             out2: paddle.Tensor,
             out3: paddle.Tensor,
             out: paddle.Tensor,
             target: paddle.Tensor,
             scale: int,
             valid_mode=False) -> paddle.Tensor:

        losses = dict()
        loss1 = self.loss_func(out1, target)
        loss2 = self.loss_func(out2, target)
        loss3 = self.loss_func(out3, target)
        loss5 = self.loss_func(out, target)
        tv_loss1 = self.regular_func(out)
        tv_loss2 = self.regular_func(out3)

        losses['loss'] = \
            0.1 * loss1 + \
            0.2 * loss2 + \
            0.3 * loss3 + \
            0.4 * loss5 + \
            0.05 * tv_loss1 +\
            0.05 * tv_loss2

        if valid_mode:
            im_h_y = out[0]  # [1,h,w]
            im_h_y = im_h_y * 255.0  # [1,h,w]
            im_h_y = im_h_y.clip(0.0, 255.0)  # [1,h,w]
            im_h_y = im_h_y[0]  # [h,w]

            im_h_y = im_h_y / 255.0  # [h,w]
            target = target.squeeze()

            rmse_predicted = self.compute_rmse(target,
                                               im_h_y,
                                               shave_border=scale)
            losses['rmse'] = rmse_predicted
        return losses

    def compute_rmse(self,
                     pred: paddle.Tensor,
                     gt: paddle.Tensor,
                     shave_border: int = 0) -> paddle.Tensor:

        height, width = pred.shape[:2]

        # shave border
        pred = pred[shave_border:height - shave_border,
                    shave_border:width - shave_border]
        gt = gt[shave_border:height - shave_border,
                shave_border:width - shave_border]

        # mask
        mask = (pred != 0).astype('float32')
        gt = mask * gt

        imdff = pred * 255.0 - gt * 255.0
        rmse = paddle.sqrt(paddle.mean(imdff**2))
        return rmse
