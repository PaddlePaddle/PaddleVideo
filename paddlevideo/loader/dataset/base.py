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

import paddle
from abc import ABC, abstractmethod
from paddle.io import Dataset
import numpy as np

import copy


class BaseDataset(Dataset, ABC):
    """Base class for datasets

    All datasets should subclass it.
    All subclass should overwrite:

    - Method: `load_file`, load info from index file.
    - Method: `prepare_train`, providing train data.
    - Method: `prepare_valid`, providing valid data.

    Args:
        file_path (str): index file path.
        pipeline (Sequence XXX)
        data_prefix (str): directory path of the data. Default: None.
        valid_mode (bool): whether to build valid dataset. Default: False.

    """
    def __init__(self, file_path, pipeline, data_prefix=None, valid_mode=False):

        super().__init__()

        self.file_path = file_path
        #        self.data_prefix = osp.realpath(data_prefix) if \
        #            osp.isdir(data_prefix) else data_prefix  #hj: data_prefix cannot be None
        self.valid_mode = valid_mode

        self.pipeline = pipeline


#        self.info = self.load_file() #hj: not do this in base class, when overwrite the method in subclass, attribute may not match

    @abstractmethod
    def load_file(self):
        """load the video information from the index file path."""
        pass

    def prepare_train(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.info[idx])
        #Note: For now, paddle.io.DataLoader cannot support dict type retval, so convert to list here
        to_list = self.pipeline(results)
        #XXX have to unsqueeze label here or before calc metric!
        return [to_list['imgs'], np.array([to_list['labels']])]

    def prepare_valid(self, idx):
        """Prepare the frames for valid given the index."""
        results = copy.deepcopy(self.info[idx])
        #Note: For now, paddle.io.DataLoader cannot support dict type retval, so convert to list here
        to_list = self.pipeline(results)
        return [to_list['imgs'], to_list['labels']]

    def __len__(self):
        """get the size of the dataset."""
        return len(self.info)

    def __getitem__(self, idx):
        """ Get the sample for either training or testing given index"""
        if self.valid_mode:
            return self.prepare_valid(idx)
        else:
            return self.prepare_train(idx)
