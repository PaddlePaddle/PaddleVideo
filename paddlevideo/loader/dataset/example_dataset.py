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


@DATASETS.register()
class ExampleDataset(BaseDataset):
    """ ExampleDataset """
    def __init__(self, pipeline, file_path=''):
        super().__init__(file_path, pipeline)

    def load_file(self):
        """load file, abstractmethod"""
        self.x = np.random.rand(100, 20, 20)
        self.y = np.random.randint(10, size=(100, 1))

    def prepare_train(self, idx):
        """Prepare the frames for training/valid given index. """
        results = {'data': self.x[idx], 'label': self.y[idx]}
        results = self.pipeline(results)
        return results['data'], results['label']

    def __len__(self):
        """get the size of the dataset."""
        return self.x.shape[0]
