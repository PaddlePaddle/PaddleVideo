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

import numpy as np
import paddle
import csv
import paddle.nn.functional as F

from .registry import METRIC
from .base import BaseMetric
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


@METRIC.register
class SkeletonMetric(BaseMetric):
    """
    Test for Skeleton based model.
    note: only support batch size = 1, single card test.

    Args:
        out_file: str, file to save test results.
    """

    def __init__(self,
                 data_size,
                 batch_size,
                 out_file='submission.csv',
                 log_interval=1,
                 top_k=5):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval)
        self.top1 = []
        self.top5 = []
        self.values = []
        self.out_file = out_file
        self.k = top_k

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        if data[0].shape[0] != outputs.shape[0]:
            num_segs = data[0].shape[1]
            batch_size = outputs.shape[0]
            outputs = outputs.reshape(
                [batch_size // num_segs, num_segs, outputs.shape[-1]])
            outputs = outputs.mean(axis=1)
        if len(data) == 2:  # data with label
            labels = data[1]
            top1 = paddle.metric.accuracy(input=outputs, label=labels, k=1)
            top5 = paddle.metric.accuracy(input=outputs, label=labels, k=self.k)
            if self.world_size > 1:
                top1 = paddle.distributed.all_reduce(
                    top1, op=paddle.distributed.ReduceOp.SUM) / self.world_size
                top5 = paddle.distributed.all_reduce(
                    top5, op=paddle.distributed.ReduceOp.SUM) / self.world_size
            self.top1.append(top1.numpy())
            self.top5.append(top5.numpy())
        else:  # data without label, only support batch_size=1. Used for fsd-10.
            prob = F.softmax(outputs)
            clas = paddle.argmax(prob, axis=1).numpy()[0]
            self.values.append((batch_id, clas))

        # preds ensemble
        if batch_id % self.log_interval == 0:
            logger.info("[TEST] Processing batch {}/{} ...".format(
                batch_id,
                self.data_size // (self.batch_size * self.world_size)))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        if self.top1:  # data with label
            logger.info('[TEST] finished, avg_acc1= {}, avg_acc5= {}'.format(
                np.mean(np.array(self.top1)), np.mean(np.array(self.top5))))
        else:
            headers = ['sample_index', 'predict_category']
            with open(
                    self.out_file,
                    'w',
            ) as fp:
                writer = csv.writer(fp)
                writer.writerow(headers)
                writer.writerows(self.values)
            logger.info("Results saved in {} !".format(self.out_file))
