"""
prop_net
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
import ast

try:
    import cPickle as pickle
except BaseException:
    import pickle

import logging
import numpy as np
import paddle.fluid as fluid

import models
import reader
from models.bmn.bmn_utils import accumulate_infer_results_video
from utils import process_result

logger = logging.getLogger(__name__)


class ModelProp(object):
    """
    model_prop
    """
    def __init__(self, infer_config, model_name):
        self.model_name = model_name

        self.weight_path = infer_config[model_name.upper()]['weight_path']
        if self.weight_path:
            assert os.path.exists(
                self.weight_path), "Given weight dir {} not exist.".format(
                    self.weight_path)

        self.infer_model = models.get_model(model_name,
                                            infer_config,
                                            mode='infer')
        logger.info(self.model_name)
        logger.info(self.weight_path)

        self.place = fluid.CUDAPlace(
            0) if infer_config.COMMON.use_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)

        self.main_program = fluid.Program()
        self.start_program = fluid.Program()

        with fluid.program_guard(self.main_program, self.start_program):
            with fluid.unique_name.guard():
                self.new_scope = fluid.Scope()
                with fluid.scope_guard(self.new_scope):
                    self.infer_model.build_input(use_dataloader=False)
                    self.infer_model.build_model()
                    self.infer_feeds = self.infer_model.feeds()
                    self.infer_outputs = self.infer_model.outputs()
                    # restore model
                    self.infer_model.load_test_weights(self.exe,
                                                       self.weight_path,
                                                       self.main_program,
                                                       self.place)
                    #self.infer_model.load_var_params(self.exe, self.weight_path, self.main_program, self.place)

        self.infer_feeder = fluid.DataFeeder(place=self.place,
                                             feed_list=self.infer_feeds)

    def predict_TSN(self, infer_config, material=None):
        """
        predict_TSN
        """
        self.fetch_list = self.infer_model.fetches()
        infer_reader = reader.get_reader(self.model_name.upper(),
                                         'infer',
                                         infer_config,
                                         material=material)
        feature_list = []
        for infer_iter, data in enumerate(infer_reader()):
            data_feed_in = [items[:-1] for items in data]

            with fluid.program_guard(self.main_program, self.start_program):
                with fluid.scope_guard(self.new_scope):
                    infer_outs = self.exe.run(
                        fetch_list=self.fetch_list,
                        feed=self.infer_feeder.feed(data_feed_in))

            feature = np.squeeze(np.array(infer_outs[1]))
            feature_list.append(feature)
        feature_values = np.vstack(feature_list)
        return feature_values

    def predict_audio(self, infer_config, material=None):
        """
        predict_audio
        """
        self.fetch_list = self.infer_model.fetches()
        infer_reader = reader.get_reader(self.model_name.upper(),
                                         'infer',
                                         infer_config,
                                         material=material)
        feature_list = []
        for infer_iter, data in enumerate(infer_reader()):
            data_feed_in = data

            with fluid.program_guard(self.main_program, self.start_program):
                with fluid.scope_guard(self.new_scope):
                    infer_outs = self.exe.run(
                        fetch_list=self.fetch_list,
                        feed=self.infer_feeder.feed(data_feed_in))

            feature = np.squeeze(np.array(infer_outs[0]))
            feature_list.append(feature)
        feature_values = np.vstack(feature_list)
        return feature_values

    def predict_BMN(self, infer_config, material=None):
        """
        predict_BMN
        """
        self.fetch_list = self.infer_model.fetches()

        periods = []
        cur_time = time.time()

        infer_reader = reader.get_reader(self.model_name.upper(),
                                         'infer',
                                         infer_config,
                                         material=material)

        for infer_iter, data in enumerate(infer_reader()):
            data_feed_in = [items[:-2] for items in data]
            video_wind_batch = [items[-2] for items in data]
            feature_info = [items[-1] for items in data]
            # print(" test  ", infer_iter, video_wind_batch)
            # print(" test  ", infer_iter, feature_info)
            feature_T = feature_info[0][0]
            feature_N = feature_info[0][1]
            with fluid.program_guard(self.main_program, self.start_program):
                with fluid.scope_guard(self.new_scope):
                    infer_outs = self.exe.run(
                        fetch_list=self.fetch_list,
                        feed=self.infer_feeder.feed(data_feed_in))

            prev_time = cur_time
            cur_time = time.time()
            period = cur_time - prev_time
            periods.append(period)

            infer_result_list = [item
                                 for item in infer_outs] + [video_wind_batch]
            if infer_iter == 0:
                video_pred_bmn_sum = np.zeros([1, 2, feature_N, feature_T])
                video_pred_start_sum = np.zeros([1, feature_T])
                video_pred_end_sum = np.zeros([1, feature_T])
                video_pred_cnt = np.zeros([1, feature_T])
            for video_wind in video_wind_batch:
                video_pred_bmn_sum[:, :, :, video_wind[0]: video_wind[1]] = infer_result_list[0] + \
                    video_pred_bmn_sum[:, :, :,
                                       video_wind[0]:video_wind[1]]
                video_pred_start_sum[:, video_wind[0]: video_wind[1]] = infer_result_list[1] + \
                    video_pred_start_sum[:,
                                         video_wind[0]:video_wind[1]]
                video_pred_end_sum[:, video_wind[0]: video_wind[1]] = infer_result_list[2] + \
                    video_pred_end_sum[:, video_wind[0]:video_wind[1]]
                video_pred_cnt[:, video_wind[0]: video_wind[1]] = np.ones([1, video_wind[1] - video_wind[0]]) + \
                    video_pred_cnt[:, video_wind[0]:video_wind[1]]

        video_pred_bmn_mean = video_pred_bmn_sum / video_pred_cnt
        video_pred_start_mean = video_pred_start_sum / video_pred_cnt
        video_pred_end_mean = video_pred_end_sum / video_pred_cnt
        video_infer_result_list = [
            video_pred_bmn_mean, video_pred_start_mean, video_pred_end_mean
        ]
        # print("BMN finish")
        score_vector_list = accumulate_infer_results_video(
            video_infer_result_list)

        # print("csv finish")
        min_frame_thread = infer_config.COMMON.fps
        nms_thread = infer_config[self.model_name.upper()]['nms_thread']
        min_pred_score = infer_config[self.model_name.upper()]['score_thread']
        nms_result = process_result.process_video_prop(score_vector_list, min_frame_thread, \
                                                       nms_thread, min_pred_score)

        bmn_res = []
        for res in nms_result:
            bmn_res.append({'start': res[0], 'end': res[1], 'score': res[2]})
        return bmn_res

    def predict_action(self, infer_config, material=None):
        """
        predict_action
        """
        self.fetch_list = [x.name for x in self.infer_outputs]
        periods = []
        results = []
        cur_time = time.time()
        infer_reader = reader.get_reader(self.model_name.upper(),
                                         'infer',
                                         infer_config,
                                         material=material)
        for infer_iter, data in enumerate(infer_reader()):
            data_feed_in = [items[:-2] for items in data]
            video_id = [[items[-2], items[-1]] for items in data]
            with fluid.program_guard(self.main_program, self.start_program):
                with fluid.scope_guard(self.new_scope):
                    infer_outs = self.exe.run(
                        fetch_list=self.fetch_list,
                        feed=self.infer_feeder.feed(data_feed_in))

            predictions_id = np.array(infer_outs[0])
            predictions_iou = np.array(infer_outs[1])
            for i in range(len(predictions_id)):
                topk_inds = predictions_id[i].argsort(
                )[0 - infer_config.COMMON.clssify_topk:]
                topk_inds = topk_inds[::-1]
                preds_id = predictions_id[i][topk_inds]
                preds_iou = predictions_iou[i][0]
                results.append((video_id[i], preds_id.tolist(),
                                topk_inds.tolist(), preds_iou.tolist()))
            prev_time = cur_time
            cur_time = time.time()
            period = cur_time - prev_time
            periods.append(period)
        logger.info('[INFER] infer finished. average time: {}'.format( \
            np.mean(periods)))

        label_map_file = infer_config.COMMON.label_dic
        fps = infer_config.COMMON.fps
        frame_offset = infer_config[self.model_name.upper()]['nms_offset']
        nms_thread = infer_config[self.model_name.upper()]['nms_thread']
        nms_id = 5

        classify_score_thread = infer_config[
            self.model_name.upper()]['classify_score_thread']
        iou_score_thread = infer_config[
            self.model_name.upper()]['iou_score_thread']
        predict_result = process_result.get_action_result(
            results, label_map_file, fps, classify_score_thread,
            iou_score_thread, nms_id, nms_thread, frame_offset)
        return predict_result
