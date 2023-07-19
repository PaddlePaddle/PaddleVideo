"""
utils
"""
#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import traceback
import logging
import shutil

import numpy as np
import paddle
import paddle.static as static
import static as static


logger = logging.getLogger(__name__)


def test_with_pyreader(exe,
                       compiled_test_prog,
                       test_pyreader,
                       test_fetch_list,
                       test_metrics,
                       log_interval=0):
    """test_with_pyreader
    """
    if not test_pyreader:
        logger.error("[TEST] get pyreader failed.")
    test_metrics.reset()
    test_iter = 0
    label_all = []
    pred_all = []
    try:
        for data in test_pyreader():
            test_outs = exe.run(compiled_test_prog,
                                fetch_list=test_fetch_list,
                                feed=data)
            loss = np.array(test_outs[0])
            pred = np.array(test_outs[1])
            label = np.array(test_outs[-1])
            pred_all.extend(pred)
            label_all.extend(label)
            test_metrics.accumulate(loss, pred, label)
            test_iter += 1
        test_metrics.finalize_and_log_out("[TEST] Finish")
    except Exception as e:
        logger.warn(
            "[TEST] fail to execute test or calculate metrics: {}".format(e))
        traceback.print_exc()
    metrics_dict, test_loss = test_metrics.get_computed_metrics()
    metrics_dict['label_all'] = label_all
    metrics_dict['pred_all'] = pred_all
    return test_loss, metrics_dict


def train_with_pyreader(exe, train_prog, compiled_train_prog, train_pyreader,
                        train_fetch_list, train_metrics, epochs=10,
                        log_interval=0, valid_interval=0,
                        save_dir='./', save_model_name='model',
                        test_exe=None, test_pyreader=None,
                        test_fetch_list=None, test_metrics=None):
    """train_with_pyreader
    """
    if not train_pyreader:
        logger.error("[TRAIN] get pyreader failed.")
    EARLY_STOP_NUM = 20
    early_stop = EARLY_STOP_NUM
    global_iter = 0
    train_iter = 0
    iter_all = 0
    best_test_acc1 = 0

    for epoch in range(epochs):
        lr = static.global_scope().find_var("learning_rate").get_tensor()
        logger.info(
            "------- learning rate {}, learning rate counter  -----".format(
                np.array(lr)))
        if early_stop < 0:
            logger.info('Earyly Stop !!!')
            break
        train_metrics.reset()
        global_iter += train_iter
        epoch_periods = []
        for data in train_pyreader():
            try:
                cur_time = time.time()
                train_outs = exe.run(compiled_train_prog,
                                     fetch_list=train_fetch_list,
                                     feed=data)
                iter_all += 1
                period = time.time() - cur_time
                epoch_periods.append(period)
                loss = np.array(train_outs[0])
                pred = np.array(train_outs[1])
                label = np.array(train_outs[-1])
                train_metrics.accumulate(loss, pred, label)
                if log_interval > 0 and (train_iter % log_interval == 0):
                    # eval here
                    train_metrics.finalize_and_log_out(
                                info='[TRAIN] Epoch {} iter {} everage: '.format(epoch, train_iter))
                train_iter += 1
            except Exception as e:
                logger.info(
                    "[TRAIN] Epoch {}, iter {} data training failed: {}".
                    format(epoch, train_iter, str(e)))
        if len(epoch_periods) < 1:
            logger.info(
                'No iteration was executed, please check the data reader')
            sys.exit(1)

        logger.info(
            '[TRAIN] Epoch {} training finished, average time: {}'.format(
                epoch, np.mean(epoch_periods)))
        train_metrics.finalize_and_log_out( \
            info='[TRAIN] Finished ... Epoch {} all iters average: '.format(epoch))

        # save models of min loss in best acc epochs
        if test_exe and valid_interval > 0 and (epoch +
                                                1) % valid_interval == 0:
            # metrics_dict,loss = train_metrics.calculator.get_computed_metrics()
            loss, metrics_dict_test = test_with_pyreader(
                exe, test_exe, test_pyreader, test_fetch_list, test_metrics,
                log_interval)
            test_acc1 = metrics_dict_test['avg_acc1']
            if test_acc1 > best_test_acc1:
                best_test_acc1 = test_acc1
                save_model(exe, train_prog, save_dir, save_model_name,
                           "_epoch{}_acc{}".format(epoch, best_test_acc1))
                early_stop = EARLY_STOP_NUM
            else:
                early_stop -= 1


def save_model(exe, program, save_dir, model_name, postfix=None):
    """save_model
    """
    model_path = os.path.join(save_dir, model_name + postfix)
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    # fluid.io.save_persistables(exe, model_path, main_program=program)
    save_vars = [x for x in program.list_vars() \
                                 if isinstance(x, paddle.framework.Parameter)]

    static.save_vars(exe,
                       dirname=model_path,
                       main_program=program,
                       vars=save_vars,
                       filename="param")


def save_model_persist(exe, program, save_dir, model_name, postfix=None):
    """save_model"""
    model_path = os.path.join(save_dir, model_name + postfix)
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    paddle.fluid.io.save_persistables(exe,
                               save_dir,
                               main_program=program,
                               filename=model_path)


def init_pretraining_params(exe,
                            pretraining_params_path,
                            main_program,
                            use_fp16=False):
    """
    init pretrain_params
    """
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    def existed_params(var):
        """
        Load existed params
        """
        if not isinstance(var, paddle.framework.Parameter):
            return False
        flag = os.path.exists(os.path.join(pretraining_params_path, var.name))
        return flag

    static.load_vars(exe,
                       pretraining_params_path,
                       main_program=main_program,
                       predicate=existed_params)
    logger.info(
        "Load pretraining parameters from {}.".format(pretraining_params_path))


class AttrDict(dict):
    """AttrDict
    """
    def __getattr__(self, key):
        """getter
        """
        return self[key]

    def __setattr__(self, key, value):
        """setter
        """
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value
