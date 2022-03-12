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
import os.path as osp
import random

from ...utils import get_logger
from ..registry import DATASETS
from .base import BaseDataset

logger = get_logger("paddlevideo")


@DATASETS.register()
class MatDataset(BaseDataset):
    def __init__(self,
                 file_path,
                 pipeline,
                 num_retries=5,
                 data_prefix=None,
                 test_mode=False):
        self.num_retries = num_retries
        super().__init__(file_path, pipeline, data_prefix, test_mode)

    def load_file(self):
        """Load index file to get h5 information."""
        info = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                rel_path = line.strip()
                abs_path = osp.join(self.data_prefix, rel_path)
                info.append(dict(filename=abs_path))
        return info

    def prepare_train(self, idx):
        """Prepare the frames for training/valid given index. """
        # Try to catch Exception caused by reading missing frames files
        for ir in range(self.num_retries):
            try:
                results = copy.deepcopy(self.info[idx])
                results = self.pipeline(results)
            except Exception as e:
                # logger.info(e)
                if ir < self.num_retries - 1:
                    logger.info(
                        "Error when loading {}, have {} trys, will try again".
                        format(
                            osp.join(self.data_prefix,
                                     self.info[idx]['mat_path']), ir))
                idx = random.randint(0, len(self.info) - 1)
                continue
            return results['imgs'], results['labels']

    def prepare_test(self, idx):
        """Prepare the frames for test given index. """
        # Try to catch Exception caused by reading missing frames files
        for ir in range(self.num_retries):
            try:
                results = copy.deepcopy(self.info[idx])
                results = self.pipeline(results)
            except Exception as e:
                # logger.info(e)
                if ir < self.num_retries - 1:
                    logger.info(
                        "Error when loading {}, have {} trys, will try again".
                        format(
                            osp.join(self.data_prefix,
                                     self.info[idx]['mat_path']), ir))
                idx = random.randint(0, len(self.info) - 1)
                continue
            return results['imgs'], results['labels']
