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
import random

try:
    import h5py
except ImportError as e:
    print(
        f"{e}, [h5py] package and it's dependencies is required for WAFP-Net.")
import numpy as np

from ...utils import get_logger
from ..registry import DATASETS
from .base import BaseDataset

logger = get_logger("paddlevideo")


@DATASETS.register()
class HDF5Dataset(BaseDataset):
    """Decode HDF5 data using h5py

    Args:
        BaseDataset (_type_): _description_
    """
    def __init__(self,
                 file_path,
                 pipeline,
                 num_retries=5,
                 data_prefix=None,
                 test_mode=False):
        self.num_retries = num_retries
        super().__init__(file_path, pipeline, data_prefix, test_mode)

        # NOTE: load hdf5 data in memory.
        self.load_file()

    def load_file(self):
        """Load index file to get h5 information."""
        hf = h5py.File(self.file_path, mode="r")
        logger.info(f"HDF5 Data Loaded from {self.file_path}")
        self.data = hf.get('data')
        self.target = hf.get('label')
        if self.data.shape[0] != self.target.shape[0]:
            raise ValueError(
                f"number of input data({self.data.shape[0]}) "
                f"must equals to label data({self.target.shape[0]})")
        # keep it open, or error will occurs
        # hf.close()

    def prepare_train(self, idx):
        """Prepare the frames for training/valid given index. """
        # Try to catch Exception caused by reading missing frames files
        for ir in range(self.num_retries):
            try:
                lr = copy.deepcopy(self.data[idx])
                hr = copy.deepcopy(self.target[idx])
                results = {'imgs': lr, 'labels': hr}
            except Exception as e:
                #logger.info(e)
                if ir < self.num_retries - 1:
                    logger.info(
                        "Error when loading {}, have {} trys, will try again".
                        format(self.file_path, ir))
                idx = random.randint(0, len(self) - 1)
                continue
            return np.array(results['imgs']), np.array(results['labels'])

    def prepare_test(self, idx):
        """Prepare the frames for test given index. """
        # Try to catch Exception caused by reading missing frames files
        for ir in range(self.num_retries):
            try:
                lr = copy.deepcopy(self.data[idx])
                hr = copy.deepcopy(self.target[idx])
                results = {'imgs': lr, 'labels': hr}
            except Exception as e:
                #logger.info(e)
                if ir < self.num_retries - 1:
                    logger.info(
                        "Error when loading {}, have {} trys, will try again".
                        format(self.file_path, ir))
                idx = random.randint(0, len(self) - 1)
                continue
            return np.array(results['imgs']), np.array(results['labels'])

    def __len__(self):
        """get the size of the dataset."""
        return self.data.shape[0]
