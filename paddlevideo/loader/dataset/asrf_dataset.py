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
class ASRFDataset(BaseDataset):
    """Video dataset for action segmentation.
    """

    def __init__(
        self,
        file_path,
        pipeline,
        feature_path,
        label_path,
        boundary_path,
        **kwargs,
    ):
        super().__init__(file_path, pipeline, **kwargs)
        self.label_path = label_path
        self.boundary_path = boundary_path
        self.feature_path = feature_path

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
        file_name = video_name.split('.')[0] + ".npy"
        label_file_path = os.path.join(self.label_path, file_name)
        label = np.load(label_file_path).astype(np.int64)

        # load boundary
        file_name = video_name.split('.')[0] + ".npy"
        boundary_file_path = os.path.join(self.boundary_path, file_name)
        boundary = np.expand_dims(np.load(boundary_file_path),axis=0).astype(np.float32)

        results['video_feat'] = copy.deepcopy(video_feat)
        results['video_label'] = copy.deepcopy(label)
        results['video_boundary'] = copy.deepcopy(boundary)

        results = self.pipeline(results)
        return results['video_feat'], results['video_label'], results['video_boundary']

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
        file_name = video_name.split('.')[0] + ".npy"
        label_file_path = os.path.join(self.label_path, file_name)
        label = np.load(label_file_path).astype(np.int64)

        # load boundary
        file_name = video_name.split('.')[0] + ".npy"
        boundary_file_path = os.path.join(self.boundary_path, file_name)
        boundary = np.expand_dims(np.load(boundary_file_path),axis=0).astype(np.float32)

        results['video_feat'] = copy.deepcopy(video_feat)
        results['video_label'] = copy.deepcopy(label)
        results['video_boundary'] = copy.deepcopy(boundary)

        results = self.pipeline(results)
        return results['video_feat'], results['video_label'], results['video_boundary']
