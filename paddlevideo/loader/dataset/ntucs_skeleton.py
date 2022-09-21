# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

from ..registry import DATASETS
from .ucf101_skeleton import UCF101SkeletonDataset
from ...utils import get_logger

logger = get_logger("paddlevideo")


@DATASETS.register()
class NTUCSSkeletonDataset(UCF101SkeletonDataset):

    def __init__(self,
                 file_path,
                 pipeline,
                 split,
                 repeat_times,
                 test_mode=False):
        super(NTUCSSkeletonDataset, self).__init__(file_path, pipeline, split,
                                                   repeat_times, test_mode)
