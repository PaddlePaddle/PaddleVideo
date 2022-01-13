# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless requifFred by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import random

import numpy as np
import paddle

from EIVideo.paddlevideo.tasks import (test_model)
from EIVideo.paddlevideo.utils import get_config, get_dist_info
from EIVideo import EI_VIDEO_ROOT, join_root_path

DEF_CONFIG_FILE_PATH = join_root_path("configs/manet.yaml")
DEF_PARAMS_FILE_PATH = join_root_path("model/default_manet.pdparams")


def parse_args():
    parser = argparse.ArgumentParser("PaddleVideo train script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default=DEF_CONFIG_FILE_PATH,
                        help='config file path')
    parser.add_argument('-o',
                        '--override',
                        action='append',
                        default=[],
                        help='config options to be overridden')
    parser.add_argument('--test',
                        action='store_true',
                        help='whether to test a model')
    parser.add_argument('--train_dali',
                        action='store_true',
                        help='whether to use dali to speed up training')
    parser.add_argument('--multigrid',
                        action='store_true',
                        help='whether to use multigrid training')
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        default=DEF_PARAMS_FILE_PATH,
                        help='weights for finetuning or testing')
    parser.add_argument('--fleet',
                        action='store_true',
                        help='whether to use fleet run distributed training')
    parser.add_argument('--amp',
                        action='store_true',
                        help='whether to open amp training.')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='fixed all random seeds when the program is running')
    parser.add_argument(
        '--max_iters',
        type=int,
        default=None,
        help='max iterations when training(this argonly used in test_tipc)')
    parser.add_argument(
        '-p',
        '--profiler_options',
        type=str,
        default=None,
        help='The option of profiler, which should be in format '
             '\"key1=value1;key2=value2;key3=value3\".')
    parser.add_argument('--use_npu',
                        type=bool,
                        default=False,
                        help='whether use npu.')

    args = parser.parse_args()
    return args


def main(**kwargs):
    args = parse_args()
    cfg = get_config(args.config, overrides=args.override)
    # ToDo To AP-kai: 下面这行代码目的是更新配置，这样的话我们调用main(use_npu = Ture)，这时cfg.use_npu就是Ture了
    for key, value in kwargs.items():
        cfg.__setattr__(key, value)

    # set seed if specified
    seed = args.seed
    if seed is not None:
        assert isinstance(
            seed,
            int), f"seed must be a integer when specified, but got {seed}"
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    _, world_size = get_dist_info()
    parallel = world_size != 1
    if parallel:
        paddle.distributed.init_parallel_env()
    final = test_model(cfg, weights=args.weights, parallel=parallel)
    return final


if __name__ == '__main__':
    main(video_path='example/example1.mp4', save_path='./output')
