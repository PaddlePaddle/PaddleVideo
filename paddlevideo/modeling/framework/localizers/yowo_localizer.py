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

from ...registry import LOCALIZERS
from .base import BaseLocalizer
from ...yowo_utils import truths_length, nms, get_region_boxes, bbox_iou

import paddle


@LOCALIZERS.register()
class YOWOLocalizer(BaseLocalizer):
    """YOWO Localization framework
    """

    def forward_net(self, imgs):
        """Call backbone forward.
        """
        # imgs.shape=[N,C,T,H,W], for YOWO
        preds = self.backbone(imgs)
        return preds

    def train_step(self, data_batch):
        """Training step.
        """
        x_data = data_batch[0]
        target = data_batch[1].squeeze(1)  # indeed do squeeze to adapt to paddle tensor
        target.stop_gradient = True

        # call Model forward
        out = self.forward_net(x_data)
        # call Loss forward
        loss, nCorrect = self.loss(out, target)
        loss_metrics = dict()
        loss_metrics['loss'] = loss
        loss_metrics['nCorrect'] = nCorrect
        return loss_metrics

    def val_step(self, data_batch):
        """Validating setp.
        """
        total = 0.0
        proposals = 0.0
        correct = 0.0
        fscore = 0.0
        eps = 1e-5
        nms_thresh = 0.4
        iou_thresh = 0.5

        x_data = data_batch[0]
        target = data_batch[1].squeeze(1)  # indeed do squeeze to adapt to paddle tensor
        frame_idx = data_batch[2]
        target.stop_gradient = True
        # call Model forward
        out = self.forward_net(x_data)
        all_boxes = get_region_boxes(out)
        out_boxes = []

        for i in range(out.shape[0]):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            out_boxes.append(boxes)
            truths = target[i].reshape([-1, 5])
            num_gts = truths_length(truths)
            total = total + num_gts
            pred_list = []
            for i in range(len(boxes)):
                if boxes[i][4] > 0.25:
                    proposals = proposals + 1
                    pred_list.append(i)
            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                best_j = -1
                for j in pred_list:  # ITERATE THROUGH ONLY CONFIDENT BOXES
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and int(boxes[best_j][6]) == box_gt[6]:
                    correct = correct + 1

        precision = 1.0 * correct / (proposals + eps)
        recall = 1.0 * correct / (total + eps)
        fscore = 2.0 * precision * recall / (precision + recall + eps)

        outs = dict()
        outs['precision'] = precision
        outs['recall'] = recall
        outs['fscore'] = fscore
        outs['frame_idx'] = frame_idx
        return outs

    def test_step(self, data_batch):
        """Test step.
        """
        total = 0.0
        proposals = 0.0
        correct = 0.0
        fscore = 0.0
        eps = 1e-5
        nms_thresh = 0.4
        iou_thresh = 0.5

        x_data = data_batch[0]
        target = data_batch[1].squeeze(1)  # indeed do squeeze to adapt to paddle tensor
        frame_idx = data_batch[2]
        target.stop_gradient = True
        # call Model forward
        out = self.forward_net(x_data)
        all_boxes = get_region_boxes(out)
        out_boxes = []

        for i in range(out.shape[0]):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            out_boxes.append(boxes)
            truths = target[i].reshape([-1, 5])
            num_gts = truths_length(truths)
            total = total + num_gts
            pred_list = []
            for i in range(len(boxes)):
                if boxes[i][4] > 0.25:
                    proposals = proposals + 1
                    pred_list.append(i)
            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                best_j = -1
                for j in pred_list:  # ITERATE THROUGH ONLY CONFIDENT BOXES
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and int(boxes[best_j][6]) == box_gt[6]:
                    correct = correct + 1

        precision = 1.0 * correct / (proposals + eps)
        recall = 1.0 * correct / (total + eps)
        fscore = 2.0 * precision * recall / (precision + recall + eps)

        outs = dict()
        outs['boxes'] = out_boxes
        outs['precision'] = precision
        outs['recall'] = recall
        outs['fscore'] = fscore
        outs['frame_idx'] = frame_idx
        return outs

    def infer_step(self, data_batch):
        """Infer step.
        """
        out = self.forward_net(data_batch[0])
        return out