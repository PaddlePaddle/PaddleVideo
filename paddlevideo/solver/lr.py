# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
from .slowfast_lr_policy import get_epoch_lr


def build_lr(cfg):
    """
    Build a learning rate scheduler accroding to ```OPTIMIZER``` configuration, and it always pass into the optimizer.

    In configuration:

    learning_rate:
        name: 'PiecewiseDecay'
        boundaries: None  # cal in lr.py
        values: None #cal in lr.py
        data_size: None #get from train.py
        max_epoch:
        warmup_epochs:
        warmup_start_lr:
        base_lr:

    Returns:
        A paddle.optimizer.lr instance.
    """

    # XXX use build?
    cfg_copy = cfg.copy()

    lr_name = cfg_copy.pop('name')

    #  get slowfast lr
    max_epoch = cfg_copy.pop('max_epoch')
    data_size = cfg_copy.pop('data_size')
    warmup_epochs = cfg_copy.pop('warmup_epochs')
    warmup_start_lr = cfg_copy.pop('warmup_start_lr')
    base_lr = cfg_copy.pop('base_lr')
    lr_list = []
    bd_list = []
    cur_bd = 1
    for cur_epoch in range(max_epoch):
        for cur_iter in range(data_size):
            cur_lr = get_epoch_lr(cur_epoch + float(cur_iter) / data_size,
                                  warmup_epochs, warmup_start_lr, base_lr,
                                  max_epoch)
            lr_list.append(cur_lr)
            bd_list.append(cur_bd)
            cur_bd += 1
    bd_list.pop()

    cfg_copy['boundaries'] = bd_list
    cfg_copy['values'] = lr_list
    #########

    return getattr(paddle.optimizer.lr, lr_name)(**cfg_copy)
