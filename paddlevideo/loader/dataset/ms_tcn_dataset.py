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

import os
import numpy as np

from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger

logger = get_logger("paddlevideo")


@DATASETS.register()
class MSTCNDataset(BaseDataset):
    """Video dataset for action segmentation.
    """

    def __init__(
        self,
        file_path,
        pipeline,
        feature_path,
        gt_path,
        actions_map_file_path,
        **kwargs,
    ):
        super().__init__(file_path, pipeline, **kwargs)
        self.gt_path = gt_path
        self.actions_map_file_path = actions_map_file_path
        self.feature_path = feature_path

        # actions dict generate
        file_ptr = open(self.actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])

        self.num_classes = len(self.actions_dict.keys())

    def load_file(self):
        """Load index file to get video information."""
        file_ptr = open(self.file_path, 'r')
        info = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        return info

    def prepare_train(self, idx):
        """TRAIN & VALID: Prepare data for training/valid given the index."""
        results = {}
        video_name = self.info[idx]
        # load video feature
        file_name = video_name.split('.')[0] + ".npy"
        feat_file_path = os.path.join(self.feature_path, file_name)
        #TODO: check path
        video_feat = np.load(feat_file_path)

        # load label
        target_file_path = os.path.join(self.gt_path, video_name)
        file_ptr = open(target_file_path, 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros(min(np.shape(video_feat)[1], len(content)), dtype='int64')
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
        # classes = classes * (-100)

        results['video_feat'] = copy.deepcopy(video_feat)
        results['video_gt'] = copy.deepcopy(classes)

        results = self.pipeline(results)
        return results['video_feat'], results['video_gt']

    def prepare_test(self, idx):
        """TEST: Prepare the data for test given the index."""
        results = {}
        video_name = self.info[idx]
        # load video feature
        file_name = video_name.split('.')[0] + ".npy"
        feat_file_path = os.path.join(self.feature_path, file_name)
        #TODO: check path
        video_feat = np.load(feat_file_path)

        # load label
        target_file_path = os.path.join(self.gt_path, video_name)
        file_ptr = open(target_file_path, 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros(min(np.shape(video_feat)[1], len(content)))
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
        # classes = classes * (-100)

        results['video_feat'] = copy.deepcopy(video_feat)
        results['video_gt'] = copy.deepcopy(classes)

        results = self.pipeline(results)
        return results['video_feat'], results['video_gt']
