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

import copy
import paddle

def build_lr(cfg):
    """
    Build a learning rate scheduler accroding to ```OPTIMIZER``` configuration, and it always pass into the optimizer.

    In configuration:

    learning_rate:
        name: 'PiecewiseDecay'
        boundaries: [20, 60]
        values: [0.00025, 0.000025, 0.0000025]
 
    
    Returns:
        A paddle.optimizer.lr instance.
    """

    # XXX use build?
    cfg_copy = cfg.copy()

    lr_name = cfg_copy.pop('name')
    
    return getattr(paddle.optimizer.lr, lr_name)(**cfg_copy)
