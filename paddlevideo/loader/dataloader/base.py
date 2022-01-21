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
from abc import ABC, abstractmethod
from paddle.io import DataLoader


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_sampler=None, **kwargs):
        super(BaseDataLoader, self).__init__(dataset, **kwargs)
        if batch_sampler is not None:
            self.batch_sampler.sampler = batch_sampler
