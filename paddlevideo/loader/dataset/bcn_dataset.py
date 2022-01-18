# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import copy
import json
import paddle
import paddle.nn.functional as F
import numpy as np

from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger

logger = get_logger("paddlevideo")


@DATASETS.register()
class BcnBgmDataset(BaseDataset):
    """Video dataset for BCN bgm model.
    """

    def __init__(
        self,
        file_path,
        pipeline,
        # mode,
        use_full,
        bd_ratio=0.05,
        **kwargs,
    ):
        super().__init__(file_path, pipeline, **kwargs)

        # assert parameter
        # assert mode in ['train', 'test'], "mode parameter must be 'train' or 'test'"
        assert use_full in [True,
                            False], "use_full parameter must be True or False"
        assert '//' not in file_path, "don't use '//' in file_path, please use '/'"

        # set parameter
        self.boundary_ratio = bd_ratio
        # self.mode = mode
        self.use_full = use_full

        # get other parameter from file_path
        file_path_list = file_path.split('/')
        root = '/'.join(file_path_list[:-2]) + '/'

        self.dataset = file_path_list[-3]
        self.gt_path = root + 'groundTruth/'
        self.features_path = root + 'features/'
        mapping_file = root + 'mapping.txt'
        file_ptr = open(mapping_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])
        # self.num_classes = len(actions_dict)

        # see mapping.txt for details
        if self.dataset == '50salads':
            self.bg_class = [17, 18]  # background
            self.resized_temporal_scale = 400
            self.sample_rate = 2
        elif self.dataset == 'gtea':
            self.boundary_ratio = 0.1
            self.bg_class = [10]
            self.resized_temporal_scale = 300  # 100 in bcn-torch
            self.sample_rate = 1
        elif self.dataset == 'breakfast':
            self.bg_class = [0]
            self.resized_temporal_scale = 300
            self.sample_rate = 1

        # get all data_path
        self.file_path = file_path
        file_ptr = open(file_path, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

    def load_file(self):
        """Load index file to get video information."""
        file_ptr = open(self.file_path, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        return self.list_of_examples

    def prepare_train(self, idx):
        """TRAIN & VALID: Prepare data for training/valid given the index."""
        feature_tensor, target_tensor, anchor_xmin, anchor_xmax = self._get_base_data(
            idx)
        result = dict()
        result['feature_tensor'] = feature_tensor
        result['target_tensor'] = target_tensor
        result['anchor_xmin'] = anchor_xmin
        result['anchor_xmax'] = anchor_xmax
        result['idx'] = idx
        result['pipeline_parameter'] = {
            'use_full': self.use_full,
            'resized_temporal_scale': self.resized_temporal_scale,
            'bg_class': self.bg_class,
            'boundary_ratio': self.boundary_ratio
        }
        result['video_name'] = self.list_of_examples[idx]
        return self.pipeline(result)

    def prepare_test(self, idx):
        """TEST: Prepare the data for test given the index."""

        return self.prepare_train(idx)

    def _get_base_data(self, index):
        """Get base data for dataset."""
        features = np.load(self.features_path +
                           self.list_of_examples[index].split('.')[0] + '.npy')
        features = copy.deepcopy(features)
        file_ptr = open(self.gt_path + self.list_of_examples[index], 'r')
        content = file_ptr.read().split('\n')[:-1]  # read ground truth
        content = copy.deepcopy(content)

        # initialize and produce gt vector
        classes = np.zeros(min(np.shape(features)[1], len(content)))
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]

        # sample information by skipping each sample_rate frames
        features = features[:, ::self.sample_rate]
        feature_tensor = paddle.to_tensor(features, dtype='float32')
        temporal_scale = feature_tensor.shape[1]
        temporal_gap = 1.0 / temporal_scale
        if self.use_full == False:
            num_frames = np.shape(features)[1]
            feature_tensor = feature_tensor.unsqueeze(0)
            if self.dataset == 'breakfast':  # for breakfast dataset, there are extremely short videos
                factor = 1
                while factor * num_frames < self.resized_temporal_scale:
                    factor = factor + 1
                feature_tensor = F.interpolate(feature_tensor,
                                               scale_factor=(factor),
                                               mode='linear',
                                               align_corners=False,
                                               data_format='NCW')
            feature_tensor = F.interpolate(feature_tensor.unsqueeze(3),
                                           size=(self.resized_temporal_scale,
                                                 1),
                                           mode='nearest').squeeze(3)
            feature_tensor = feature_tensor.squeeze(0)
            temporal_scale = self.resized_temporal_scale
            temporal_gap = 1.0 / temporal_scale
        target = classes[::self.sample_rate]
        target_tensor = paddle.to_tensor(target, dtype='int64')
        anchor_xmin = [temporal_gap * i for i in range(temporal_scale)]
        anchor_xmax = [temporal_gap * i for i in range(1, temporal_scale + 1)]

        return feature_tensor, target_tensor, anchor_xmin, anchor_xmax

    def __len__(self):
        return len(self.list_of_examples)


@DATASETS.register()
class BcnModelDataset(BaseDataset):
    """Video dataset for BCN main model.
    """

    def __init__(
        self,
        file_path,
        pipeline,
        # mode,
        bd_ratio=0.05,
        **kwargs,
    ):
        super().__init__(file_path, pipeline, **kwargs)

        # assert parameter
        assert '//' not in file_path, "don't use '//' in file_path, please use '/'"

        # set parameter
        self.boundary_ratio = bd_ratio

        # get other parameter from file_path
        file_path_list = file_path.split('/')
        root = '/'.join(file_path_list[:-2]) + '/'

        self.dataset = file_path_list[-3]
        self.gt_path = root + 'groundTruth/'
        self.features_path = root + 'features/'
        mapping_file = root + 'mapping.txt'
        file_ptr = open(mapping_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])
        self.num_classes = len(self.actions_dict)

        # see mapping.txt for details
        if self.dataset == '50salads':
            self.bg_class = [17, 18]
            self.sample_rate = 2
        elif self.dataset == 'gtea':
            self.boundary_ratio = 0.1
            self.bg_class = [10]
            self.sample_rate = 1
        elif self.dataset == 'breakfast':
            self.bg_class = [0]
            self.sample_rate = 1

        # get all data_path
        self.index = 0
        self.file_path = file_path
        file_ptr = open(file_path, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

    def load_file(self):
        """Load index file to get video information."""
        file_ptr = open(self.file_path, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        return self.list_of_examples

    def prepare_train(self, idx):
        """TRAIN & VALID: Prepare data for training/valid given the index."""
        feature_tensor, target_tensor, mask, anchor_xmin, anchor_xmax = self._get_base_data(
            idx)
        match_score_start, match_score_end = self._get_train_label(
            idx, target_tensor, anchor_xmin, anchor_xmax)
        match_score = paddle.concat(
            (match_score_start.unsqueeze(0), match_score_end.unsqueeze(0)), 0)
        match_score = paddle.max(match_score, 0)  #.values()
        result = dict()
        result['feature_tensor'] = feature_tensor
        result['target_tensor'] = target_tensor
        result['mask'] = mask
        result['match_score'] = match_score
        result['video_name'] = self.list_of_examples[idx]
        return result

    def prepare_test(self, idx):
        """TEST: Prepare the data for test given the index."""

        return self.prepare_train(idx)

    def __len__(self):
        return len(self.list_of_examples)

    def _get_base_data(self, index):
        """Get base data for dataset."""
        features = np.load(self.features_path +
                           self.list_of_examples[index].split('.')[0] + '.npy')
        file_ptr = open(self.gt_path + self.list_of_examples[index], 'r')
        content = file_ptr.read().split('\n')[:-1]  # read ground truth
        # initialize and produce gt vector
        classes = np.zeros(min(np.shape(features)[1], len(content)))
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]

        # sample information by skipping each sample_rate frames
        features = features[:, ::self.sample_rate]
        target = classes[::self.sample_rate]

        # create pytorch tensor
        feature_tensor = paddle.to_tensor(features)
        feature_tensor = paddle.cast(feature_tensor, 'float32')
        target_tensor = paddle.to_tensor(target)
        target_tensor = paddle.cast(target_tensor, 'int64')
        mask = paddle.ones([self.num_classes, np.shape(target)[0]])
        mask = paddle.cast(mask, 'float32')

        total_frame = target_tensor.shape[0]
        temporal_scale = total_frame
        temporal_gap = 1.0 / temporal_scale
        anchor_xmin = [temporal_gap * i for i in range(temporal_scale)]
        anchor_xmax = [temporal_gap * i for i in range(1, temporal_scale + 1)]
        return feature_tensor, target_tensor, mask, anchor_xmin, anchor_xmax

    def _get_train_label(self, index, target_tensor, anchor_xmin, anchor_xmax):
        """Process base data to get train label."""
        total_frame = target_tensor.shape[0]
        temporal_scale = total_frame
        temporal_gap = 1.0 / temporal_scale
        gt_label, gt_starts, gt_ends = self._get_labels_start_end_time(
            target_tensor, self.bg_class)  # original length
        gt_label, gt_starts, gt_ends = np.array(gt_label), np.array(
            gt_starts), np.array(gt_ends)
        gt_starts, gt_ends = gt_starts.astype(np.float64), gt_ends.astype(
            np.float64)
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
        """Calculate score"""
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.0)
        scores = np.divide(inter_len, len_anchors)
        return scores

    def _get_labels_start_end_time(self, target_tensor, bg_class):
        """Get labels clip:[label, start time, end time]"""
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
