# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import math
from paddle.optimizer.lr import *
"""
PaddleVideo Learning Rate Schedule:
You can use paddle.optimizer.lr
or define your custom_lr in this file.
"""


class CustomWarmupCosineDecay(LRScheduler):
    r"""
    We combine warmup and stepwise-cosine which is used in slowfast model.

    Args:
        warmup_start_lr (float): start learning rate used in warmup stage.
        warmup_epochs (int): the number epochs of warmup.
        cosine_base_lr (float|int, optional): base learning rate in cosine schedule.
        max_epoch (int): total training epochs.
        num_iters(int): number iterations of each epoch.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .
    Returns:
        ``CosineAnnealingDecay`` instance to schedule learning rate.
    """
    def __init__(self,
                 warmup_start_lr,
                 warmup_epochs,
                 cosine_base_lr,
                 max_epoch,
                 num_iters,
                 last_epoch=-1,
                 verbose=False):
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.cosine_base_lr = cosine_base_lr
        self.max_epoch = max_epoch
        self.num_iters = num_iters
        #call step() in base class, last_lr/last_epoch/base_lr will be update
        super(CustomWarmupCosineDecay, self).__init__(last_epoch=last_epoch,
                                                      verbose=verbose)

    def step(self, epoch=None):
        """
        ``step`` should be called after ``optimizer.step`` . It will update the learning rate in optimizer according to current ``epoch`` .
        The new learning rate will take effect on next ``optimizer.step`` .
        Args:
            epoch (int, None): specify current epoch. Default: None. Auto-increment from last_epoch=-1.
        Returns:
            None
        """
        if epoch is None:
            if self.last_epoch == -1:
                self.last_epoch += 1
            else:
                self.last_epoch += 1 / self.num_iters  # update step with iters
        else:
            self.last_epoch = epoch
        self.last_lr = self.get_lr()

        if self.verbose:
            print('Epoch {}: {} set learning rate to {}.'.format(
                self.last_epoch, self.__class__.__name__, self.last_lr))

    def _lr_func_cosine(self, cur_epoch, cosine_base_lr, max_epoch):
        return cosine_base_lr * (math.cos(math.pi * cur_epoch / max_epoch) +
                                 1.0) * 0.5

    def get_lr(self):
        """Define lr policy"""
        lr = self._lr_func_cosine(self.last_epoch, self.cosine_base_lr,
                                  self.max_epoch)
        lr_end = self._lr_func_cosine(self.warmup_epochs, self.cosine_base_lr,
                                      self.max_epoch)

        # Perform warm up.
        if self.last_epoch < self.warmup_epochs:
            lr_start = self.warmup_start_lr
            alpha = (lr_end - lr_start) / self.warmup_epochs
            lr = self.last_epoch * alpha + lr_start
        return lr


class CustomPiecewiseDecay(PiecewiseDecay):
    def __init__(self, **kargs):
        kargs.pop('num_iters')
        super().__init__(**kargs)
