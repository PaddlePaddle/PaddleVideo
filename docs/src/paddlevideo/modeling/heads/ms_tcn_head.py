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
import paddle.nn.functional as F
import numpy as np

from paddle import ParamAttr

from .base import BaseHead
from ..registry import HEADS
from ..weight_init import weight_init_


@HEADS.register()
class MSTCNHead(BaseHead):

    def __init__(self, num_classes, in_channels):
        super().__init__(num_classes, in_channels)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

        # cls score
        self.overlap = 0.5

    def forward(self, x):
        """MS-TCN no head
        """
        return x

    def loss(self, output, video_gt):
        """calculate loss
        """
        output_transpose = paddle.transpose(output, [2, 0, 1])
        ce_x = paddle.reshape(output_transpose,
                              (output_transpose.shape[0] *
                               output_transpose.shape[1], self.num_classes))
        ce_y = video_gt[0, :]
        ce_loss = self.ce(ce_x, ce_y)
        loss = ce_loss

        mse = self.mse(F.log_softmax(output[:, :, 1:], axis=1),
                       F.log_softmax(output.detach()[:, :, :-1], axis=1))
        mse = paddle.clip(mse, min=0, max=16)
        mse_loss = 0.15 * paddle.mean(mse)
        loss += mse_loss

        return loss

    def get_F1_score(self, predicted, groundTruth):
        recog_content = list(predicted.numpy())
        gt_content = list(groundTruth[0].numpy())

        # cls score
        correct = 0
        total = 0
        edit = 0

        for i in range(len(gt_content)):
            total += 1

            if gt_content[i] == recog_content[i]:
                correct += 1

        edit_num = self.edit_score(recog_content, gt_content)
        edit += edit_num

        tp, fp, fn = self.f_score(recog_content, gt_content, self.overlap)

        # cls metric

        precision = tp / float(tp + fp)
        recall = tp / float(fp + fn)

        if precision + recall > 0.0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1 = np.nan_to_num(f1)
        return f1

    def get_labels_start_end_time(self, frame_wise_labels):
        labels = []
        starts = []
        ends = []
        last_label = frame_wise_labels[0]
        labels.append(frame_wise_labels[0])
        starts.append(0)
        for i in range(len(frame_wise_labels)):
            if frame_wise_labels[i] != last_label:
                labels.append(frame_wise_labels[i])
                starts.append(i)
                ends.append(i)
                last_label = frame_wise_labels[i]
        ends.append(i + 1)
        return labels, starts, ends

    def levenstein(self, p, y, norm=False):
        m_row = len(p)
        n_col = len(y)
        D = np.zeros([m_row + 1, n_col + 1], np.float)
        for i in range(m_row + 1):
            D[i, 0] = i
        for i in range(n_col + 1):
            D[0, i] = i

        for j in range(1, n_col + 1):
            for i in range(1, m_row + 1):
                if y[j - 1] == p[i - 1]:
                    D[i, j] = D[i - 1, j - 1]
                else:
                    D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1,
                                  D[i - 1, j - 1] + 1)

        if norm:
            score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
        else:
            score = D[-1, -1]

        return score

    def edit_score(self, recognized, ground_truth, norm=True):
        P, _, _ = self.get_labels_start_end_time(recognized)
        Y, _, _ = self.get_labels_start_end_time(ground_truth)
        return self.levenstein(P, Y, norm)

    def f_score(self, recognized, ground_truth, overlap):
        p_label, p_start, p_end = self.get_labels_start_end_time(recognized)
        y_label, y_start, y_end = self.get_labels_start_end_time(ground_truth)

        tp = 0
        fp = 0

        hits = np.zeros(len(y_label))

        for j in range(len(p_label)):
            intersection = np.minimum(p_end[j], y_end) - np.maximum(
                p_start[j], y_start)
            union = np.maximum(p_end[j], y_end) - np.minimum(
                p_start[j], y_start)
            IoU = (1.0 * intersection / union) * (
                [p_label[j] == y_label[x] for x in range(len(y_label))])
            # Get the best scoring segment
            idx = np.array(IoU).argmax()

            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1
        fn = len(y_label) - sum(hits)
        return float(tp), float(fp), float(fn)
