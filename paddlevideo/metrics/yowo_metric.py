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

import os.path as osp
import os
from paddlevideo.utils import get_logger
from .registry import METRIC
from .base import BaseMetric
from .ucf24_utils import get_mAP

logger = get_logger("paddlevideo")


@METRIC.register
class YOWOMetric(BaseMetric):
    """
    Metrics for YOWO. Two Stages in this metric:
    (1) Get test results using trained model, results will be saved in YOWOMetric.result_path;
    (2) Calculate metrics using results file from stage (1).
    """

    def __init__(self,
                 data_size,
                 batch_size,
                 gt_folder,
                 result_path,
                 threshold=0.5,
                 save_path=None,
                 log_interval=1):
        """
        Init for BMN metrics.
        Params:
            gtfolder:groundtruth folder path for ucf24
        """
        super().__init__(data_size, batch_size, log_interval)
        self.result_path = result_path
        self.gt_folder = gt_folder
        self.threshold = threshold
        self.save_path = save_path

        if not osp.isdir(self.result_path):
            os.makedirs(self.result_path)

    def update(self, batch_id, data, outputs):
        frame_idx = outputs['frame_idx']
        boxes = outputs["boxes"]
        for j in range(len(frame_idx)):
            detection_path = osp.join(self.result_path, frame_idx[j])
            with open(detection_path, 'w+') as f_detect:
                for box in boxes[j]:
                    x1 = round(float(box[0] - box[2] / 2.0) * 320.0)
                    y1 = round(float(box[1] - box[3] / 2.0) * 240.0)
                    x2 = round(float(box[0] + box[2] / 2.0) * 320.0)
                    y2 = round(float(box[1] + box[3] / 2.0) * 240.0)

                    det_conf = float(box[4])
                    for j in range((len(box) - 5) // 2):
                        cls_conf = float(box[5 + 2 * j].item())
                        prob = det_conf * cls_conf
                        f_detect.write(
                            str(int(box[6]) + 1) + ' ' + str(prob) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(
                                x2) + ' ' + str(y2) + '\n')
        if batch_id % self.log_interval == 0:
            logger.info("[TEST] Processing batch {}/{} ...".format(
                batch_id,
                self.data_size // (self.batch_size * self.world_size)))

    def accumulate(self):
        metric_list = get_mAP(self.gt_folder, self.result_path, self.threshold, self.save_path)
        for info in metric_list:
            logger.info(info)
