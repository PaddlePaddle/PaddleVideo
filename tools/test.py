# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import os
import sys
import argparse
import paddle


__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from paddlevideo.utils import get_config
from paddlevideo.loader.builder import build_dataloader, build_dataset
from paddlevideo.modeling.builder import build_model
from paddlevideo.tasks import test_model
from paddlevideo.utils import get_dist_info


def parse_args():
    parser = argparse.ArgumentParser("PaddleVideo test script")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/recognition/tsm/tsm.yaml',
        help='config file path')
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    parser.add_argument(
        '-w',
        '--weight',
        default='',
        help='weight path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = get_config(args.config)

    dataset = build_dataset((cfg.DATASET.test, cfg.PIPELINE.test))
    _, world_size = get_dist_info()
    parallel = world_size != 1
    if parallel:
        paddle.distributed.init_parallel_env()

    model = build_model(cfg.MODEL)

    test_model(model, dataset, cfg, weight, parallel)


if __name__ == '__main__':
    main()


