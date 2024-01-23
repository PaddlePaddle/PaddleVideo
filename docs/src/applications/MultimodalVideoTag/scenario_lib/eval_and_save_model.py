"""
eval main
"""
#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import os
import sys
import time
import argparse
import logging
import pickle

import numpy as np
import paddle
paddle.enable_static()
import paddle.static as static

from accuracy_metrics import MetricsCalculator
from datareader import get_reader
from config import parse_config, merge_configs, print_configs
from models.attention_lstm_ernie import AttentionLstmErnie
from utils import test_with_pyreader


def parse_args():
    """parse_args
    """
    parser = argparse.ArgumentParser("Paddle Video evaluate script")
    parser.add_argument('--model_name',
                        type=str,
                        default='BaiduNet',
                        help='name of model to train.')
    parser.add_argument('--config',
                        type=str,
                        default='configs/conf.txt',
                        help='path to config file of model')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help=
        'path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument('--output', type=str, default=None, help='output path')
    parser.add_argument('--use_gpu',
                        type=bool,
                        default=True,
                        help='default use gpu.')
    parser.add_argument('--save_model_param_dir',
                        type=str,
                        default=None,
                        help='checkpoint path')
    parser.add_argument('--save_inference_model',
                        type=str,
                        default=None,
                        help='save inference path')
    parser.add_argument('--save_only',
                        action='store_true',
                        default=False,
                        help='only save model, do not evaluate model')
    args = parser.parse_args()
    return args


def evaluate(args):
    """evaluate
    """
    # parse config
    config = parse_config(args.config)
    valid_config = merge_configs(config, 'valid', vars(args))
    print_configs(valid_config, 'Valid')

    # build model
    valid_model = AttentionLstmErnie(args.model_name,
                                     valid_config,
                                     mode='valid')
    startup = static.Program()
    valid_prog = static.default_main_program().clone(for_test=True)
    with static.program_guard(valid_prog, startup):
        paddle.disable_static()
        valid_model.build_input(True)
        valid_model.build_model()
        valid_feeds = valid_model.feeds()
        valid_outputs = valid_model.outputs()
        valid_loss = valid_model.loss()
        valid_pyreader = valid_model.pyreader()
        paddle.enable_static()

    place = paddle.CUDAPlace(0) if args.use_gpu else paddle.CPUPlace()
    exe = static.Executor(place)
    exe.run(startup)
    compiled_valid_prog = static.CompiledProgram(valid_prog)

    # load weights
    assert os.path.exists(args.save_model_param_dir), \
            "Given save weight dir {} not exist.".format(args.save_model_param_dir)
    valid_model.load_test_weights_file(exe, args.save_model_param_dir,
                                       valid_prog, place)

    if args.save_inference_model:
        save_model_params(exe, valid_prog, valid_model,
                          args.save_inference_model)

    if args.save_only is True:
        print('save model only, exit')
        return

    # get reader
    bs_denominator = 1
    valid_config.VALID.batch_size = int(valid_config.VALID.batch_size /
                                        bs_denominator)
    valid_reader = get_reader(args.model_name.upper(), 'valid', valid_config)

    # get metrics
    valid_metrics = MetricsCalculator(args.model_name.upper(), 'valid',
                                      valid_config)
    valid_fetch_list = [valid_loss.name] + [x.name for x in valid_outputs
                                            ] + [valid_feeds[-1].name]
    # get reader
    exe_places = static.cuda_places() if args.use_gpu else static.cpu_places()
    valid_pyreader.decorate_sample_list_generator(valid_reader,
                                                  places=exe_places)

    test_loss, metrics_dict_test = test_with_pyreader(exe, compiled_valid_prog,
                                                      valid_pyreader,
                                                      valid_fetch_list,
                                                      valid_metrics)
    test_acc1 = metrics_dict_test['avg_acc1']
    print(test_loss)
    print(test_acc1)


def save_model_params(exe, program, model_object, save_dir):
    """save_model_params
    """
    feeded_var_names = [var.name for var in model_object.feeds()][:-1]
    static.save_inference_model(dirname=save_dir,
                                  feeded_var_names=feeded_var_names,
                                  main_program=program,
                                  target_vars=model_object.outputs(),
                                  executor=exe,
                                  model_filename='model',
                                  params_filename='params')

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
