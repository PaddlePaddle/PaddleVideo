"""
train main
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


import numpy as np
import paddle
paddle.enable_static()
import paddle.static as static

from accuracy_metrics import MetricsCalculator
from datareader import get_reader
from config import print_configs, merge_configs, parse_config
from models.attention_lstm_ernie import AttentionLstmErnie
from utils import init_pretraining_params, train_with_pyreader


logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    """parse_args
    """
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='BaiduNet',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/conf.txt',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='path to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--use_gpu', type=bool, default=True, help='default use gpu.')
    parser.add_argument(
        '--no_use_pyreader',
        action='store_true',
        default=False,
        help='whether to use pyreader')
    parser.add_argument(
        '--no_memory_optimize',
        action='store_true',
        default=False,
        help='whether to use memory optimize in train')
    parser.add_argument(
        '--epoch_num',
        type=int,
        default=0,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--valid_interval',
        type=int,
        default=1,
        help='validation epoch interval, 0 for no validation.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints',
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--save_log_name',
        type=str,
        default='train_val',
        help='save to tensorboard filename recommand model name.')
    args = parser.parse_args()
    return args


def train(args):
    """train main
    """
    # parse config
    config = parse_config(args.config)
    train_config = merge_configs(config, 'train', vars(args))
    valid_config = merge_configs(config, 'valid', vars(args))
    print_configs(train_config, 'Train')
    train_model = AttentionLstmErnie(args.model_name, train_config, mode='train')
    valid_model = AttentionLstmErnie(args.model_name, valid_config, mode='valid')

    max_train_steps = train_config.TRAIN.epoch * train_config.TRAIN.num_samples // train_config.TRAIN.batch_size
    print('max train steps %d' % (max_train_steps))
    # build model
    startup = static.Program()
    train_prog = static.Program()
    with static.program_guard(train_prog, startup):
        paddle.disable_static()
        train_model.build_input(use_pyreader=True)
        train_model.build_model()
            # for the input, has the form [data1, data2,..., label], so train_feeds[-1] is label
        train_feeds = train_model.feeds()
        train_feeds[-1].persistable = True
            # for the output of classification model, has the form [pred]
        train_outputs = train_model.outputs()
        for output in train_outputs:
            output.persistable = True
        train_loss = train_model.loss()
        train_loss.persistable = True
            # outputs, loss, label should be fetched, so set persistable to be true
        optimizer = train_model.optimizer()
        optimizer.minimize(train_loss)
        train_pyreader = train_model.pyreader()
        paddle.enable_static()

    if not args.no_memory_optimize:
        paddle.distributed.transpiler.memory_optimize(train_prog)

    valid_prog = static.Program()
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

    if args.resume:
        # if resume weights is given, load resume weights directly
        assert os.path.exists(args.resume), \
            "Given resume weight dir {} not exist.".format(args.resume)

        def if_exist(var):
            """if_exist
            """
            return os.path.exists(os.path.join(args.resume, var.name))

        print('resuming ,,,,,,,,,,,,,,')
        paddle.fluid.io.load_persistables(
                    exe, '', main_program=train_prog, filename=args.resume)

    else:
        # load ernie pretrain model
        init_pretraining_params(exe,
                                train_config.TRAIN.ernie_pretrain_dict_path,
                                main_program=train_prog)
        # if not in resume mode, load pretrain weights
        # this pretrain may be only audio or video
        if args.pretrain:
            assert os.path.exists(args.pretrain), \
                "Given pretrain weight dir {} not exist.".format(args.pretrain)
        if args.pretrain:
            train_model.load_test_weights_file(exe, args.pretrain, train_prog, place)

    build_strategy = paddle.static.BuildStrategy()
    build_strategy.enable_inplace = True

    compiled_train_prog = static.CompiledProgram(
        train_prog).with_data_parallel(loss_name=train_loss.name,
                                       build_strategy=build_strategy)
    compiled_valid_prog = static.CompiledProgram(
        valid_prog).with_data_parallel(share_vars_from=compiled_train_prog,
                                       build_strategy=build_strategy)

    # get reader
    bs_denominator = 1
    if (not args.no_use_pyreader) and args.use_gpu:
        dev_list = static.cuda_places()
        bs_denominator = len(dev_list)
    train_config.TRAIN.batch_size = int(train_config.TRAIN.batch_size /
                                        bs_denominator)
    valid_config.VALID.batch_size = int(valid_config.VALID.batch_size /
                                        bs_denominator)
    train_reader = get_reader(args.model_name.upper(), 'train', train_config)
    valid_reader = get_reader(args.model_name.upper(), 'valid', valid_config)

    exe_places = static.cuda_places() if args.use_gpu else static.cpu_places()
    train_pyreader.decorate_sample_list_generator(train_reader,
                                                  places=exe_places)
    valid_pyreader.decorate_sample_list_generator(valid_reader,
                                                  places=exe_places)

    # get metrics
    train_metrics = MetricsCalculator(args.model_name.upper(), 'train', train_config)
    valid_metrics = MetricsCalculator(args.model_name.upper(), 'valid', valid_config)
    # print("****************************valid_metrics", valid_metrics.get())
    train_fetch_list = [train_loss.name] + [x.name for x in train_outputs
                                            ] + [train_feeds[-1].name]
    valid_fetch_list = [valid_loss.name] + [x.name for x in valid_outputs
                                            ] + [valid_feeds[-1].name]

    epochs = args.epoch_num or train_model.epoch_num()

    train_with_pyreader(
        exe,
        train_prog,
        compiled_train_prog,
        train_pyreader,
        train_fetch_list,
        train_metrics,
        epochs=epochs,
        log_interval=args.log_interval,
        valid_interval=args.valid_interval,
        save_dir=args.save_dir,
        save_model_name=args.model_name,
        test_exe=compiled_valid_prog,
        test_pyreader=valid_pyreader,
        test_fetch_list=valid_fetch_list,
        test_metrics=valid_metrics)

if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train(args)
