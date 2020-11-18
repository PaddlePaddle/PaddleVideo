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
import os.path as osp
import copy
import numpy as np

from ..registry import DATASETS
from .base import BaseDataset 

@DATASETS.register()
class FrameDataset(BaseDataset):
    """Rawframe dataset for action recognition.
    The dataset loads raw frames from frame files, and apply specified transform operatation them.
    The indecx file is a text file with multiple lines, and each line indicates the directory of frames of a video, toatl frames of the video, and its label, which split with a whitespace.
    Example of an index file:

    .. code-block:: txt

        file_path-1 150 1
        file_path-2 160 1
        file_path-3 170 2
        file_path-4 180 2

    Args:
        file_path (str): Path to the index file.
        pipeline(XXX):
        data_prefix (str): directory path of the data. Default: None.
        valid_mode (bool): Whether to bulid the valid dataset. Default: False.
        suffix (str): suffix of file. Default: 'img_{:05}.jpg'.

    """
    def __init__(self,
                 file_path,
                 pipeline,
                 data_prefix=None,
                 valid_mode=False,
                 suffix='img_{:05}.jpg'):

        #unique attribute in frames dataset.
        self.suffix = suffix

        super().__init__(
                 file_path,
                 pipeline,
                 data_prefix,
                 valid_mode)

    def load_file(self):
        """Load index file to get video information."""
        info = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                frame_dir, frames_len, labels = line_split
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                info.append(dict(frame_dir=frame_dir, frames_len=frames_len, labels=int(labels)))
        return info

    def prepare_train(self, idx):
        """Prepare the frames for training given index. """
        results = copy.deepcopy(self.info[idx])
        results['suffix'] = self.suffix
        #Note: For now, paddle.io.DataLoader cannot support dict type retval, so convert to list here
        to_list =  self.pipeline(results)
        #XXX have to unsqueeze label here or before calc metric!
        return [to_list['imgs'], np.array([to_list['labels']])]


    def prepare_valid(self, idx):
        """Prepare the frames for training given index. """
        results = copy.deepcopy(self.info[idx])
        results['suffix'] = self.suffix
        #Note: For now, paddle.io.DataLoader cannot support dict type retval, so convert to list here
        to_list =  self.pipeline(results)
        #XXX have to unsqueeze label here or before calc metric!
        return [to_list['imgs'], np.array([to_list['labels']])]

