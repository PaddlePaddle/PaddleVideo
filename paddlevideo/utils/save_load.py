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
import paddle.nn.functional as F
from paddlevideo.utils import get_logger
from paddlevideo.utils import main_only


#XXX(shipping): maybe need load N times because of different cards have different params.
@main_only
def load_ckpt(model,
              weight_path,
              num_patches=None,
              seg_num=None,
              attention_type=None):
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
        if "VisionTransformer" in str(model):  # For TimeSformer case
            tmp = state_dicts
        else:  # For 2DCNN case
            for item in tqdm(model.state_dict(), total=total_len, position=0):
                name = item
                desc.set_description('Loading %s' % name)
                if name not in state_dicts:
                    if str('backbone.' + name) in state_dicts:
                        tmp[name] = state_dicts['backbone.' + name]
                else:  # Common case
                    tmp[name] = state_dicts[name]
                time.sleep(0.01)

        if "VisionTransformer" in str(model):  # For TimeSformer case
            if 'head' + '.weight' in tmp:
                del tmp['head' + '.weight']
            if 'head' + '.bias' in tmp:
                del tmp['head' + '.bias']

            logger.info("Loading %s" % 'pos_embed')
            if num_patches + 1 != tmp['pos_embed'].shape[1]:
                pos_embed = tmp['pos_embed']
                cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
                other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).unsqueeze(
                    1).transpose((0, 1, 3, 2))
                new_pos_embed = F.interpolate(other_pos_embed,
                                              size=(other_pos_embed.shape[-2],
                                                    num_patches),
                                              mode='nearest')
                new_pos_embed = new_pos_embed.squeeze(0).transpose((0, 2, 1))
                new_pos_embed = paddle.concat((cls_pos_embed, new_pos_embed), 1)
                tmp['pos_embed'] = new_pos_embed
                time.sleep(0.01)

            if 'time_embed' in tmp and seg_num != tmp['time_embed'].size(1):
                logger.info("Loading %s" % 'time_embed')
                time_embed = tmp['time_embed'].transpose((0, 2, 1)).unsqueeze(0)
                new_time_embed = F.interpolate(time_embed,
                                               size=(time_embed.shape[-2],
                                                     seg_num),
                                               mode='nearest')
                tmp['time_embed'] = new_time_embed.squeeze(0).transpose(
                    (0, 2, 1))
                time.sleep(0.01)

            if attention_type == 'divided_space_time':  # Transformer case
                new_state_dict = tmp.copy()
                for key in tmp:
                    if 'blocks' in key and 'attn' in key:
                        logger.info("Loading %s" % key)
                        new_key = key.replace('attn', 'temporal_attn')
                        if not new_key in tmp:
                            new_state_dict[new_key] = tmp[key]
                        else:
                            new_state_dict[new_key] = tmp[new_key]
                    if 'blocks' in key and 'norm1' in key:
                        logger.info("Loading %s" % key)
                        new_key = key.replace('norm1', 'temporal_norm1')
                        if not new_key in tmp:
                            new_state_dict[new_key] = tmp[key]
                        else:
                            new_state_dict[new_key] = tmp[new_key]
                tmp = new_state_dict
                time.sleep(0.01)

        ret_str = "loading {:<20d} weights completed.".format(
            len(model.state_dict()))
        desc.set_description(ret_str)

        model.set_state_dict(tmp)


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
    if not osp.isfile(file_name):
        raise IOError(f'{file_name} not exist')
    return paddle.load(file_name)
