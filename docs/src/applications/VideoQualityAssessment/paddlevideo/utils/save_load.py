"""
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
"""

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
    load_ckpt
    """
    #model.set_state_dict(state_dict)

    if not osp.isfile(weight_path):
        raise IOError('{weight_path} is not a checkpoint file')
    #state_dicts = load(weight_path)

    logger = get_logger("paddlevideo")
    state_dicts = paddle.load(weight_path)
    tmp = {}
    total_len = len(model.state_dict())
    localkeyname = [i for i in state_dicts]

    with tqdm(total=total_len,
              position=1,
              bar_format='{desc}',
              desc="Loading weights") as desc:
        #for item in tqdm(model.state_dict(), total=total_len, position=0):
        for i, item in enumerate(
                tqdm(model.state_dict(), total=total_len, position=0)):
            name = item
            desc.set_description('Loading %s' % name)
            print("model name is {}, correspoding local name is {}".format(
                name, localkeyname[i]))
            #tmp[name] = state_dicts[name]
            tmp[name] = state_dicts[localkeyname[i]]
            time.sleep(0.01)
        ret_str = "loading {:<20d} weights completed.".format(
            len(model.state_dict()))
        desc.set_description(ret_str)
        model.set_state_dict(tmp)


def mkdir(dir):
    """mkdir"""
    if not os.path.exists(dir):
        # avoid error when train with multiple gpus
        try:
            os.makedirs(dir)
        except:
            pass


@main_only
def save(obj, path):
    """save"""
    paddle.save(obj, path)


def load(file_name):
    """load"""
    if not osp.isfile(file_name):
        raise IOError('{file_name} not exist')
    return paddle.load(file_name)
