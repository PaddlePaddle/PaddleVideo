# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import pandas as pd
from scipy import signal
import os
import copy
import numpy as np

from .base import BaseHead
from ..registry import HEADS
from ..weight_init import weight_init_


@HEADS.register()
class BcnBgmHead(BaseHead):
    """
    Head for Bcn bgm model.
    Args:
        just for test.
    """

    def __init__(self,
                 use_full,
                 test_mode,
                 results_path,
                 dataset,
                 in_channels=-1,
                 num_classes=-1,
                 **kwargs):
        super().__init__(num_classes, in_channels, **kwargs)
        assert test_mode in ['less', 'more'], "test_mode must be less or more"

        self.use_full = use_full
        self.test_mode = test_mode

        if not os.path.exists(results_path):
            os.makedirs(results_path)
        self.results_path = results_path
        if dataset == 'breakfast' or dataset == 'gtea':
            self.temporal_dim = 300
        elif dataset == '50salads':
            self.temporal_dim = 400

    def forward(self, outputs, video_name):
        """don't need any parameter, just process result and save
        """
        outputs = copy.deepcopy(outputs)
        outputs = outputs.cpu().detach().numpy()
        columns = ["barrier"]
        if self.use_full:
            barrier_threshold = 0.5
            barrier = (outputs > barrier_threshold) * outputs
            video_result = barrier[0]

            video_result = video_result.transpose([1, 0])
            video_df = pd.DataFrame(list(video_result), columns=columns)
            video_df.to_csv(os.path.join(self.results_path,
                                         video_name + ".csv"),
                            index=False)

        else:
            if self.test_mode == 'less':
                barrier_threshold = 0.5
                barrier = (outputs > barrier_threshold) * outputs
                video_result = barrier[0]

                maximum = signal.argrelmax(video_result[0])
                flag = np.array([0] * self.temporal_dim)
                flag[maximum] = 1

                video_result = video_result * flag
                video_df = pd.DataFrame(list(video_result.transpose([1, 0])),
                                        columns=columns)
                video_df.to_csv(os.path.join(self.results_path,
                                             video_name + ".csv"),
                                index=False)
            elif self.test_mode == 'more':
                barrier = (outputs > 0.3) * outputs
                high_barrier = (outputs > 0.8)
                video_result = barrier[0]
                maximum1 = signal.argrelmax(video_result[0])
                maximum2 = high_barrier[0]

                flag = np.array([0] * self.temporal_dim)
                flag[maximum1] = 1
                flag = np.clip((flag + maximum2), 0, 1)

                video_result = video_result * flag
                video_df = pd.DataFrame(list(video_result.transpose([1, 0])),
                                        columns=columns)
                video_df.to_csv(os.path.join(self.results_path,
                                             video_name + ".csv"),
                                index=False)

        return None  # just process and save, don't need return
