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

import copy
import numpy as np

from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger

logger = get_logger("paddlevideo")


@DATASETS.register()
class UCF24Dataset(BaseDataset):
    """Dataset for YOWO
       The dataset loads raw videos and apply specified transforms on them.
       The index file is a file with multiple lines, and each line indicates
       a sample video with the filepath and label, which are split with a whitesapce.
       Example of a inde file:
       .. code-block:: txt

       Args:
           file_path(str): Path to the index file.
           pipeline(XXX): A sequence of data transforms.
           **kwargs: Keyword arguments for ```BaseDataset```.
    """

    def __init__(self, file_path, pipeline, num_retries=5, **kwargs):
        self.num_retries = num_retries
        super().__init__(file_path, pipeline, **kwargs)

    def load_file(self):
        """Load index file to get video information."""
        info = []
        with open(self.file_path, 'r') as fin:
            lines = fin.readlines()
        for line in lines:
            line = line.strip()  # 'data/ucf24/labels/class_name/video_name/key_frame.txt'
            filename = line.replace('txt', 'jpg').replace(
                'labels', 'rgb-images')  # key frame path

            info.append(dict(filename=filename))
        return info

    def prepare_train(self, idx):
        """TRAIN & VALID. Prepare the data for training/valid given the index."""
        results = copy.deepcopy(self.info[idx])
        results = self.pipeline(results)
        im_path = results['filename']
        im_path = im_path.replace('jpg', 'txt')
        im_split = im_path.split('/')
        frame_index = im_split[3] + '_' + im_split[4] + '_' + im_split[5]
        return results['imgs'], np.array([results['labels']]), frame_index

    def prepare_test(self, idx):
        """TEST. Prepare the data for test given the index."""
        # Try to catch Exception caused by reading corrupted video file
        results = copy.deepcopy(self.info[idx])
        results = self.pipeline(results)
        im_path = results['filename']
        im_path = im_path.replace('jpg', 'txt')
        im_split = im_path.split('/')
        frame_index = im_split[3] + '_' + im_split[4] + '_' + im_split[5]
        return results['imgs'], np.array([results['labels']]), frame_index
