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

import inspect
from typing import Dict

import paddle
from paddle.optimizer.lr import LRScheduler
from paddle.regularizer import L1Decay, L2Decay
from paddlevideo.utils import get_logger


def build_optimizer(cfg: Dict,
                    lr_scheduler: LRScheduler,
                    model: paddle.nn.Layer,
                    use_amp: bool = False,
                    amp_level: str = None) -> paddle.optimizer.Optimizer:
    """Build an optimizer and learning rate scheduler to optimize parameters accroding to ```OPTIMIZER``` field in configuration.

    In configuration:
    OPTIMIZER:
        name: Momentum
        momentum: 0.9
        weight_decay: 0.001
    or

    OPTIMIZER:
        name: Momentum
        momentum: 0.9
        weight_decay:
            name: "L1"
            value: 0.001

    Momentum optimizer will be applied to optimize network and L1Decay regularizer will be applied to avoid overfit.

    OPTIMIZER:
        name: Adam
        weight_decay:
            name: "L2"
            value: 0.001

    Adam optimizer will be applied to optimize network and L2Decay regularizer will applied to avoid overfit.

    Refer to ```https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/regularizer/L2Decay_en.html``` for more details.

    Args:
        cfg (Dict): optimizer configuration.
        lr_scheduler (LRScheduler): learning rate scheduler.
        model (paddle.nn.Layer, optional): model which contains parameters to be optimized. Defaults to None.
        use_amp (bool, optional): Whether use amp. Defaults to False.
        amp_level (str, optional): amp level when amp is enabled. Defaults to None.


    Returns:
        paddle.optimizer.Optimizer: an optimizer for the input model.
    """
    logger = get_logger("paddlevideo")
    cfg_copy = cfg.copy()
    # NOTE: check none and illegal cfg!!!
    opt_name = cfg_copy.pop('name')
    # deal with weight decay
    if cfg_copy.get('weight_decay'):
        if isinstance(cfg_copy.get('weight_decay'),
                      float):  # just an float factor
            cfg_copy['weight_decay'] = cfg_copy.get('weight_decay')
        elif 'L1' in cfg_copy.get('weight_decay').get(
                'name').upper():  # specify L2 wd and it's float factor
            cfg_copy['weight_decay'] = L1Decay(
                cfg_copy.get('weight_decay').get('value'))
        elif 'L2' in cfg_copy.get('weight_decay').get(
                'name').upper():  # specify L1 wd and it's float factor
            cfg_copy['weight_decay'] = L2Decay(
                cfg_copy.get('weight_decay').get('value'))
        else:
            raise ValueError

    # deal with grad clip
    if cfg_copy.get('grad_clip'):
        if isinstance(cfg_copy.get('grad_clip'), float):
            cfg_copy['grad_clip'] = cfg_copy.get('grad_clip').get('value')
        elif 'global' in cfg_copy.get('grad_clip').get('name').lower():
            cfg_copy['grad_clip'] = paddle.nn.ClipGradByGlobalNorm(
                cfg_copy.get('grad_clip').get('value'))
        else:
            raise ValueError

    # Set for optimizers that cannot be applied to l2decay, i.e. AdamW
    if cfg_copy.get('no_weight_decay_name'):
        no_weight_decay_name = cfg_copy.pop('no_weight_decay_name')
        no_weight_decay_name_list = no_weight_decay_name.split(' ')

        # NOTE: use param.name not name
        no_weight_decay_param_list = [
            param.name for name, param in model.named_parameters()
            if any(key_word in name for key_word in no_weight_decay_name_list)
        ]  # get the full param name of no weight decay

        _apply_decay_param_fun = lambda name: name not in no_weight_decay_param_list
        cfg_copy['apply_decay_param_fun'] = _apply_decay_param_fun
        logger.info(
            f"No weight Decay list :({len(no_weight_decay_param_list)})",
            no_weight_decay_param_list)

    cfg_copy.pop('learning_rate')

    # set multi_precision
    optimizer_setting = {
        'learning_rate': lr_scheduler,
        'parameters': model.parameters(),
        **cfg_copy
    }
    optimizer_init_args = inspect.getargspec(
        getattr(paddle.optimizer, opt_name).__init__).args
    if use_amp and amp_level == "O2" and "multi_precision" in optimizer_init_args:
        # support "multi_precision" arg in optimizer's __init__ function.
        optimizer_setting.update({"multi_precision": True})
        logger.info(
            "Set multi_precision=True for optimizer when use_amp=True and amp_level='O2'"
        )

    return getattr(paddle.optimizer, opt_name)(**optimizer_setting)
