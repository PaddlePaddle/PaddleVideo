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
             out1: paddle.Tensor = None,
             out2: paddle.Tensor = None,
             out3: paddle.Tensor = None,
             out: paddle.Tensor = None,
             target: paddle.Tensor = None,
             scale: int = 0,
             valid_mode: bool = False) -> Dict:

        losses = dict()
        if not valid_mode:
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
        else:
            loss5 = self.loss_func(out, target)
            losses['loss'] = loss5

            height, width = out.shape[-2:]
            shave_border = scale

            # shave border
            out = out[:, :, shave_border:height - shave_border,
                      shave_border:width - shave_border]
            target = target[:, :, shave_border:height - shave_border,
                            shave_border:width - shave_border]

            # mask
            mask = (out != 0).astype('float32')
            target = mask * target

            rmse_predicted = self._compute_rmse(target, out)
            losses['rmse'] = rmse_predicted
        return losses

    def _compute_rmse(self, imgs1: paddle.Tensor,
                      imgs2: paddle.Tensor) -> paddle.Tensor:

        imdff = imgs1 * 255.0 - imgs2 * 255.0
        rmse = paddle.sqrt(paddle.mean(imdff**2))
        return rmse