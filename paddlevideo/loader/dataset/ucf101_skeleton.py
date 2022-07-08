# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import os.path as osp
import copy
import random
import numpy as np
import pickle

import paddle
from paddle.io import Dataset

from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger

logger = get_logger("paddlevideo")


@DATASETS.register()
class UCF101SkeletonDataset(BaseDataset):
    """
    Skeleton dataset for action recognition.
    The dataset loads skeleton feature, and apply norm operatations.
    Args:
        file_path (str): Path to the index file.
        pipeline(obj): Define the pipeline of data preprocessing.
        test_mode (bool): Whether to bulid the test dataset. Default: False.
    """

    def __init__(self,
                 file_path,
                 pipeline,
                 split,
                 repeat_times,
                 test_mode=False):
        self.split = split
        self.repeat_times = repeat_times
        super().__init__(file_path, pipeline, test_mode=test_mode)
        self._ori_len = len(self.info)
        self.start_index = 0
        self.modality = "Pose"

    def load_file(self):
        """Load annotation file to get video information."""
        assert self.file_path.endswith('.pkl')
        return self.load_pkl_annotations()

    def load_pkl_annotations(self):
        with open(self.file_path, "rb") as f:
            data = pickle.load(f)

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            data = [x for x in data if x[identifier] in split[self.split]]

        return data

    def prepare_train(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.info[idx % self._ori_len])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        return self.pipeline(results)

    def prepare_test(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.info[idx % self._ori_len])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        return self.pipeline(results)

    def __len__(self):
        """get the size of the dataset."""
        return len(self.info) * self.repeat_times
