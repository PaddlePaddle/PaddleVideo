# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

from abc import abstractmethod

import paddle
from paddlevideo.utils import get_dist_info

from .registry import METRIC


class BaseMetric(object):
    def __init__(self, data_size, batch_size, log_interval=1, **kwargs):
        self.data_size = data_size
        self.batch_size = batch_size
        _, self.world_size = get_dist_info()
        self.log_interval = log_interval

    def gather_from_gpu(self,
                        gather_object: paddle.Tensor,
                        concat_axis=0) -> paddle.Tensor:
        """gather Tensor from all gpus into a list and concatenate them on `concat_axis`.

        Args:
            gather_object (paddle.Tensor): gather object Tensor
            concat_axis (int, optional): axis for concatenation. Defaults to 0.

        Returns:
            paddle.Tensor: gatherd & concatenated Tensor
        """
        gather_object_list = []
        paddle.distributed.all_gather(gather_object_list, gather_object)
        return paddle.concat(gather_object_list, axis=concat_axis)

    @abstractmethod
    def update(self):
        raise NotImplementedError(
            "'update' method must be implemented in subclass")

    @abstractmethod
    def accumulate(self):
        raise NotImplementedError(
            "'accumulate' method must be implemented in subclass")
