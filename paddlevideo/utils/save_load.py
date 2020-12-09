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
import os
import os.path as osp
import time

import pickle
from tqdm import tqdm
import paddle

from paddlevideo.utils import get_logger
from paddlevideo.utils import main_only


#XXX(shipping): maybe need load N times because of different cards have different params.
@main_only
def load_ckpt(model, weight_path):
    """
    """
    #model.set_state_dict(state_dict)

    if not osp.isfile(weight_path):
        raise IOError(f'{weight_path} is not a checkpoint file')
    #state_dicts = load(weight_path)

    logger = get_logger("paddlevideo")
    state_dicts = paddle.load(weight_path)
    tmp = {}
    total_len = len(model.state_dict())
    with tqdm(total=total_len,
              position=1,
              bar_format='{desc}',
              desc="Loading weights") as desc:
        for item in tqdm(model.state_dict(), total=total_len, position=0):
            name = item
            desc.set_description('Loading %s' % name)
            tmp[name] = state_dicts[name]
            time.sleep(0.01)
        ret_str = "loading {:<20d} weights completed.".format(
            len(model.state_dict()))
        desc.set_description(ret_str)
        model.set_state_dict(tmp)

#XXX(shipping): maybe need load N times because of different cards have different params.
@main_only
def load_pretrain_pptsm(model, weight_path):
    """
    """
    #load param from pretrained r50(for improve)
    assert os.path.exists(weight_path), "Given dir {} not exist.".format(weight_path)
    pre_state_dict = paddle.load(weight_path)
    param_state_dict = {}
    model_dict = model.state_dict()
    for key in model_dict.keys():
        weight_name = model_dict[key].name
        if weight_name in pre_state_dict.keys() and weight_name != "fc_0.w_0" and weight_name != "fc_0.b_0":
            param_state_dict[key] = pre_state_dict[weight_name]
        else:
            param_state_dict[key] = model_dict[key]
    model.set_dict(param_state_dict)

def mkdir(dir):
    if not os.path.exists(dir):
        # avoid error when train with multiple gpus
        try:
            os.makedirs(dir)
        except:
            pass


"""
def save(state_dicts, file_name):
    def convert(state_dict):
        model_dict = {}

        for k, v in state_dict.items():
            if isinstance(
                    v,
                (paddle.fluid.framework.Variable, paddle.fluid.core.VarBase)):
                model_dict[k] = v.numpy()
            else:
                model_dict[k] = v

        return model_dict

    final_dict = {}
    for k, v in state_dicts.items():
        if isinstance(
                v,
            (paddle.fluid.framework.Variable, paddle.fluid.core.VarBase)):
            final_dict = convert(state_dicts)
            break
        elif isinstance(v, dict):
            final_dict[k] = convert(v)
        else:
            final_dict[k] = v

    with open(file_name, 'wb') as f:
        pickle.dump(final_dict, f, protocol=2)
"""


@main_only
def save(obj, path):
    paddle.save(obj, path)


def load(file_name):
    with open(file_name, 'rb') as f:
        state_dicts = pickle.load(f, encoding='latin1')
    return state_dicts
