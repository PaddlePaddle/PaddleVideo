"""
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
"""

import numpy as np
import paddle
from paddle.hapi.model import _all_gather

from scipy import stats

from .registry import METRIC
from .base import BaseMetric
from paddlevideo.utils import get_logger
logger = get_logger("paddlevideo")


@METRIC.register
class QuqlityMetric(BaseMetric):
    """CenterCropQualityMetric"""
    def __init__(self, data_size, batch_size, log_interval=1):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval)
        self.output = []
        self.label = []
        self.y_pred = np.zeros(data_size)
        self.y_test = np.zeros(data_size)

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        labels = data[1]
        
        predict_output = paddle.tolist(outputs)
        predict_label = paddle.tolist(labels)
        predict_output_len = len(predict_output)
        for i in range(predict_output_len):
            self.output.append(predict_output[i][0])
            self.label.append(predict_label[i][0])
        
        if batch_id % self.log_interval == 0:
            logger.info("[TEST] Processing batch {}/{} ...".format(
                batch_id,
                self.data_size // (self.batch_size * self.world_size)))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        test_output_np = np.array(self.output)
        test_label_np = np.array(self.label)
        PLCC = stats.pearsonr(test_output_np, test_label_np)[0]
        SROCC = stats.spearmanr(test_output_np, test_label_np)[0]
        
        logger.info('[TEST] finished, PLCC= {}, SROCC= {} '.format(PLCC, SROCC))

    def accumulate_train(self, output, label):
        """accumulate_train"""
        output_np = np.array(output)
        label_np = np.array(label)
        PLCC = stats.pearsonr(output_np, label_np)[0]
        SROCC = stats.spearmanr(output_np, label_np)[0]
        return PLCC, SROCC

