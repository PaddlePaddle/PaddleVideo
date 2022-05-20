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

import argparse
import os
import os.path as osp
import sys

import numpy as np
import paddle
from paddleslim.quant import quant_post_static

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from paddlevideo.loader.builder import build_dataloader, build_dataset
from paddlevideo.utils import get_config, get_logger


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser("PaddleVideo Inference model script")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default=
        '../../configs/recognition/pptsm/pptsm_k400_frames_uniform_quantization.yaml',
        help='quantization config file path')
    parser.add_argument('-o',
                        '--override',
                        action='append',
                        default=[],
                        help='config options to be overridden')
    parser.add_argument("--use_gpu",
                        type=str2bool,
                        default=True,
                        help="whether use gpui during quantization")

    return parser.parse_args()


def post_training_quantization(cfg, use_gpu: bool = True):
    """Quantization entry

    Args:
        cfg (dict): quntization configuration.
        use_gpu (bool, optional): whether to use gpu during quantization. Defaults to True.
    """
    logger = get_logger("paddlevideo")

    place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()

    # get defined params
    batch_size = cfg.DATASET.get('batch_size', 1)
    num_workers = cfg.DATASET.get('num_workers', 0)
    inference_file_name = cfg.get('model_name', 'inference')
    inference_model_dir = cfg.get('inference_model_dir',
                                  f'./inference/{inference_file_name}')
    quant_output_dir = cfg.get('quant_output_dir',
                               osp.join(inference_model_dir, 'quant_model'))
    batch_nums = cfg.get('batch_nums', 10)

    # build dataloader for quantization, lite data is enough
    slim_dataset = build_dataset((cfg.DATASET.quant, cfg.PIPELINE.quant))
    slim_dataloader_setting = dict(batch_size=batch_size,
                                   num_workers=num_workers,
                                   places=place,
                                   drop_last=False,
                                   shuffle=False)
    slim_loader = build_dataloader(slim_dataset, **slim_dataloader_setting)

    logger.info("Build slim_loader finished")

    def sample_generator(loader):
        def __reader__():
            for indx, data in enumerate(loader):
                # must return np.ndarray, not paddle.Tensor
                videos = np.array(data[0])
                yield videos

        return __reader__

    # execute quantization in static graph mode
    paddle.enable_static()

    exe = paddle.static.Executor(place)

    logger.info("Staring Post-Training Quantization...")

    quant_post_static(executor=exe,
                      model_dir=inference_model_dir,
                      quantize_model_path=quant_output_dir,
                      sample_generator=sample_generator(slim_loader),
                      model_filename=f'{inference_file_name}.pdmodel',
                      params_filename=f'{inference_file_name}.pdiparams',
                      batch_size=batch_size,
                      batch_nums=batch_nums,
                      algo='KL')

    logger.info("Post-Training Quantization finished...")


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args.config, overrides=args.override)
    post_training_quantization(cfg, args.use_gpu)
