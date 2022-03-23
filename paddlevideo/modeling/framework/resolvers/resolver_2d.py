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

import math
from typing import Dict, Sequence, Tuple, Union

import paddle
from paddlevideo.utils import get_logger

from ...registry import RESOLVERS
from .base import BaseResolver

logger = get_logger("paddlevideo")


@RESOLVERS.register()
class Resolver2D(BaseResolver):
    """2D resolver model framework."""
    def _forward_net(
        self, imgs: paddle.Tensor
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor,
                                    paddle.Tensor]]:
        if self.backbone is not None:
            out1, out2, out3, out = self.backbone(imgs)
        else:
            raise NotImplementedError(f"backboen must exist.")
        if self.training:
            return out1, out2, out3, out
        else:
            return out

    def train_step(self, data_batch: Sequence[paddle.Tensor]) -> Dict:
        """Define how the model is going to train, from input to output.
        """
        imgs = data_batch[0]
        labels = data_batch[1]
        out1, out2, out3, out = self._forward_net(imgs)
        loss_metrics = self.head.loss(out1, out2, out3, out, labels)
        return loss_metrics

    def val_step(self, data_batch: Sequence[paddle.Tensor]) -> Dict:
        """Define how the model is going to validate, from input to output.
        """
        imgs = data_batch[0]
        labels = data_batch[1]
        if self.runtime_cfg.val.mode == 'full':
            out = self._forward_net(imgs)
        elif self.runtime_cfg.val.mode == 'patch':
            out = self._forward_net_patch(imgs, self.runtime_cfg.val.scale,
                                          self.runtime_cfg.val.patch_size)
        else:
            raise NotImplementedError(
                f"self.runtime_cfg.val.mode must be 'full' or 'patch', but got {self.runtime_cfg.val.mode}"
            )
        loss_metrics = self.head.loss(None,
                                      None,
                                      None,
                                      out,
                                      labels,
                                      self.runtime_cfg.val.scale,
                                      valid_mode=True)
        return loss_metrics

    def test_step(self, data_batch: Sequence[paddle.Tensor]) -> paddle.Tensor:
        """Define how the model is going to test, from input to output."""
        # NOTE: (shipping) when testing, the net won't call head.loss, we deal with the test processing in /paddlevideo/metrics
        imgs = data_batch[0]
        if self.runtime_cfg.test.mode == 'full':
            out = self._forward_net(imgs)
        elif self.runtime_cfg.test.mode == 'patch':
            out = self._forward_net_patch(imgs, self.runtime_cfg.test.scale,
                                          self.runtime_cfg.test.patch_size)
        else:
            raise NotImplementedError(
                f"self.runtime_cfg.test.mode must be 'full' or 'patch', but got {self.runtime_cfg.test.mode}"
            )
        return out

    def infer_step(self, data_batch: Sequence[paddle.Tensor]) -> paddle.Tensor:
        """Define how the model is going to infer, from input to output."""
        imgs = data_batch[0]
        if self.runtime_cfg.test.mode == 'full':
            out = self._forward_net(imgs)
        elif self.runtime_cfg.test.mode == 'patch':
            out = self._forward_net_patch(imgs, self.runtime_cfg.infer.scale,
                                          self.runtime_cfg.infer.patch_size)
        else:
            raise NotImplementedError(
                f"self.runtime_cfg.infer.mode must be 'full' or 'patch', but got {self.runtime_cfg.infer.mode}"
            )
        return out

    def _forward_net_patch(self, imgs: paddle.Tensor, scale: int,
                           sub_s: int) -> paddle.Tensor:
        """Get the output in the form of patch-wise prediction

        Args:
            imgs (paddle.Tensor): input imgs.

        Returns:
            paddle.Tensor: output full predictions.
        """
        h, w = imgs.shape[-2:]
        m = math.ceil(h / (sub_s))
        n = math.ceil(w / (sub_s))

        full_out = paddle.zeros_like(imgs, dtype='float32')  # [b,c,h,w]

        # process center
        for i in range(1, m):
            for j in range(1, n):
                begx = (i - 1) * sub_s - scale
                begy = (j - 1) * sub_s - scale
                endx = i * sub_s + scale
                endy = j * sub_s + scale
                if (begx < 0):
                    begx = 0
                if (begy < 0):
                    begy = 0
                if (endx > h):
                    endx = h
                if (endy > w):
                    endy = w

                im_input = imgs[:, :, begx:endx, begy:endy]
                out_patch = self._forward_net(im_input)
                out_patch = out_patch.detach()
                im_h_y = out_patch
                im_h_y = im_h_y * 255.0
                im_h_y = im_h_y.clip(0.0, 255.0)
                im_h_y = im_h_y / 255.0

                sh, sw = paddle.shape(im_h_y)[-2:]
                full_out[:, :, begx + scale:endx - scale, begy + scale:endy - scale] = \
                    im_h_y[:, :, scale:sh - scale, scale:sw - scale]

        # process edge
        for i in range(1, n):
            begx = h - sub_s - scale
            begy = (i - 1) * sub_s - scale
            endx = h
            endy = i * sub_s + scale
            if (begx < 0):
                begx = 0
            if (begy < 0):
                begy = 0
            if (endx > h):
                endx = h
            if (endy > w):
                endy = w
            im_input = imgs[:, :, begx:endx, begy:endy]
            out_patch = self._forward_net(im_input)
            out_patch = out_patch.detach()
            im_h_y = out_patch
            im_h_y = im_h_y * 255.0
            im_h_y = im_h_y.clip(0.0, 255.0)
            im_h_y = im_h_y / 255.0
            sh, sw = paddle.shape(im_h_y)[-2:]
            full_out[:, :, begx + scale:endx - scale, begy + scale:endy - scale] = \
                im_h_y[:, :, scale:sh - scale, scale:sw - scale]

        # process edge
        for i in range(1, m):
            begx = (i - 1) * sub_s - scale
            begy = w - sub_s - scale
            endx = i * sub_s + scale
            endy = w
            if (begx < 0):
                begx = 0
            if (begy < 0):
                begy = 0
            if (endx > h):
                endx = h
            if (endy > w):
                endy = w
            im_input = imgs[:, :, begx:endx, begy:endy]
            out_patch = self._forward_net(im_input)
            out_patch = out_patch.detach()
            im_h_y = out_patch
            im_h_y = im_h_y * 255.0
            im_h_y = im_h_y.clip(0.0, 255.0)
            im_h_y = im_h_y / 255.0
            sh, sw = paddle.shape(im_h_y)[-2:]
            full_out[:, :, begx + scale:endx - scale, begy + scale:endy - scale] = \
                im_h_y[:, :, scale:sh - scale, scale:sw - scale]
            im_input = im_input.detach()

        # process remain
        begx = h - sub_s - scale
        endx = h
        begy = w - sub_s - scale
        endy = w
        im_input = imgs[:, :, begx:endx, begy:endy]
        out_patch = self._forward_net(im_input)
        out_patch = out_patch.detach()
        im_h_y = out_patch
        im_h_y = im_h_y * 255.0
        im_h_y = im_h_y.clip(0.0, 255.0)
        im_h_y = im_h_y / 255.0
        sh, sw = paddle.shape(im_h_y)[-2:]
        full_out[:, :, begx + scale:endx - scale, begy + scale:endy - scale] = \
            im_h_y[:, :, scale:sh - scale, scale:sw - scale]

        return full_out
