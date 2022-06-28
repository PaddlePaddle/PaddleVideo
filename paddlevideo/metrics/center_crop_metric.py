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

from typing import List

import paddle
from paddlevideo.utils import get_logger

from .base import BaseMetric
from .registry import METRIC

logger = get_logger("paddlevideo")


@METRIC.register
class CenterCropMetric(BaseMetric):
    def __init__(self, data_size, batch_size, log_interval=1, **kwargs):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval, **kwargs)
        self.rest_data_size = data_size  # Number of samples remaining to be tested
        self.all_outputs = []
        self.all_labels = []
        self.topk = kwargs.get("topk", [1, 5])

    def update(self, batch_id: int, data: List, outputs: paddle.Tensor) -> None:
        """update metrics during each iter

        Args:
            batch_id (int): iter id of current batch.
            data (List): list of batched data, such as [inputs, labels]
            outputs (paddle.Tensor): batched outputs from model
        """
        labels = data[1]
        if self.world_size > 1:
            labels_gathered = self.gather_from_gpu(labels, concat_axis=0)
            outpus_gathered = self.gather_from_gpu(outputs, concat_axis=0)
        else:
            labels_gathered = labels
            outpus_gathered = outputs

        # Avoid resampling effects when testing with multiple cards
        labels_gathered = labels_gathered[0:min(len(labels_gathered), self.
                                                rest_data_size)]
        outpus_gathered = outpus_gathered[0:min(len(outpus_gathered), self.
                                                rest_data_size)]
        self.all_labels.append(labels_gathered)
        self.all_outputs.append(outpus_gathered)
        self.rest_data_size -= outpus_gathered.shape[0]

        # preds ensemble
        if batch_id % self.log_interval == 0:
            logger.info("[TEST] Processing batch {}/{} ...".format(
                batch_id,
                self.data_size // (self.batch_size * self.world_size)))

    def accumulate(self):
        """accumulate, compute, and show metrics when finished all iters.
        """
        self.all_outputs = paddle.concat(self.all_outputs, axis=0)
        self.all_labels = paddle.concat(self.all_labels, axis=0)

        result_str = []
        for _k in self.topk:
            topk_val = paddle.metric.accuracy(input=self.all_outputs,
                                              label=self.all_labels,
                                              k=_k).item()
            result_str.append(f"avg_acc{_k}={topk_val}")
        result_str = ", ".join(result_str)
        logger.info(f"[TEST] finished, {result_str}")
