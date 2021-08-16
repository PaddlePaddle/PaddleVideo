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
    def __init__(self, out_file, data_size, batch_size, log_interval=1):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval)
        self.top1 = []
        self.values = []
        self.out_file = out_file
        self.true_num = 0

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        prob = F.softmax(outputs)
        clas = paddle.argmax(prob, axis=1).numpy()[0]
        self.values.append((batch_id, clas))

        if len(data) == 2:  # data with label
            labels = data[1]
            top1 = paddle.metric.accuracy(input=outputs, label=labels, k=1)
            self.top1.append(top1.numpy())
            if clas == labels[0][0]:
                self.true_num += 1

        # preds ensemble
        if batch_id % self.log_interval == 0:
            logger.info("[TEST] Processing batch {}/{} ...".format(
                batch_id,
                self.data_size // (self.batch_size * self.world_size)))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        headers = ['sample_index', 'predict_category']
        with open(
                self.out_file,
                'w',
        ) as fp:
            writer = csv.writer(fp)
            writer.writerow(headers)
            writer.writerows(self.values)
        if self.top1:
            logger.info(
                '[TEST] finished, true_number={}, total_number={}, avg_acc1= {}'
                .format(self.true_num, len(self.values),
                        np.mean(np.array(self.top1))))
        logger.info("Results saved in {} !".format(self.out_file))
