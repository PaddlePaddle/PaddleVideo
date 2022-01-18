#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import numpy as np
import paddle
from ..registry import PIPELINES
"""pipeline ops for BCN Net.
"""


@PIPELINES.register()
class GetBcnBgmTrainLabel(object):
    """Get train label for bcn_bgm_model."""

    def __init__(self):
        pass

    def __call__(self, result):
        new_results = dict()

        # get pipeline parameter
        pipeline_parameter = result['pipeline_parameter']
        self.use_full = pipeline_parameter['use_full']
        self.resized_temporal_scale = pipeline_parameter[
            'resized_temporal_scale']
        self.bg_class = pipeline_parameter['bg_class']
        self.boundary_ratio = pipeline_parameter['boundary_ratio']

        # get train_label
        match_score_start, match_score_end = self._get_train_label(result['target_tensor'], \
                                                                result['anchor_xmin'], result['anchor_xmax'])
        match_score = paddle.concat(
            (match_score_start.unsqueeze(0), match_score_end.unsqueeze(0)), 0)
        match_score = paddle.max(match_score, 0)

        # get new_results
        new_results['feature_tensor'] = result['feature_tensor']
        new_results['match_score'] = match_score
        new_results['video_name'] = result['video_name']

        return new_results

    def _get_labels_start_end_time(self, target_tensor, bg_class):
        labels = []
        starts = []
        ends = []
        target = target_tensor.numpy()
        last_label = target[0]
        if target[0] not in bg_class:
            labels.append(target[0])
            starts.append(0)

        for i in range(np.shape(target)[0]):
            if target[i] != last_label:
                if target[i] not in bg_class:
                    labels.append(target[i])
                    starts.append(i)
                if last_label not in bg_class:
                    ends.append(i)
                last_label = target[i]

        if last_label not in bg_class:
            ends.append(np.shape(target)[0] - 1)
        return labels, starts, ends

    def _get_train_label(self, target_tensor, anchor_xmin, anchor_xmax):
        total_frame = target_tensor.shape[0]
        if self.use_full:
            temporal_gap = 1.0 / total_frame
        else:
            temporal_gap = 1.0 / self.resized_temporal_scale
        gt_label, gt_starts, gt_ends = self._get_labels_start_end_time(
            target_tensor, self.bg_class)  # original length
        gt_label, gt_starts, gt_ends = np.array(gt_label), np.array(
            gt_starts), np.array(gt_ends)
        gt_starts, gt_ends = gt_starts.astype(np.float), gt_ends.astype(
            np.float)
        gt_starts, gt_ends = gt_starts / total_frame, gt_ends / total_frame  # length to 0~1

        gt_lens = gt_ends - gt_starts
        gt_len_small = np.maximum(temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack(
            (gt_starts - gt_len_small / 2, gt_starts + gt_len_small / 2),
            axis=1)
        gt_end_bboxs = np.stack(
            (gt_ends - gt_len_small / 2, gt_ends + gt_len_small / 2), axis=1)

        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(
                np.max(
                    self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx],
                                           gt_start_bboxs[:, 0],
                                           gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(
                np.max(
                    self._ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx],
                                           gt_end_bboxs[:, 0],
                                           gt_end_bboxs[:, 1])))
        match_score_start = paddle.to_tensor(match_score_start)
        match_score_end = paddle.to_tensor(match_score_end)
        return match_score_start, match_score_end

    def _ioa_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.0)
        scores = np.divide(inter_len, len_anchors)
        return scores


@PIPELINES.register()
class BcnModelPipeline(object):
    """BCN main model do not need pipeline."""

    def __init__(self):
        pass

    def __call__(self, result):
        return result
