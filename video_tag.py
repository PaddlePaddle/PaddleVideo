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

import paddle
import argparse
from paddlevideo.utils import get_config
from paddlevideo.tasks import infer_model
from paddlevideo.utils import get_dist_info


def parse_args():
    parser = argparse.ArgumentParser("PaddleVideo train script")
    parser.add_argument('-ec',
                        '--extractor_config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument('-pc',
                        '--predictor_config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument('-o',
                        '--override',
                        action='append',
                        default=[],
                        help='config options to be overridden')

    parser.add_argument('--weights',
                        type=str,
                        help='weights for finetuning or testing')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    extractor_cfg = get_config(args.extractor_config, overrides=args.override)
    predictor_cfg = get_config(args.predictor_config, overrides=args.override)

    _, world_size = get_dist_info()
    parallel = world_size != 1
    if parallel:
        paddle.distributed.init_parallel_env()

    infer_model(extractor_cfg,
                predictor_cfg,
                weights=args.weights,
                parallel=parallel)


if __name__ == '__main__':
    main()
