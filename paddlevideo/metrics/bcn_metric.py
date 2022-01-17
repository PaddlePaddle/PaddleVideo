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

import os
import json
import numpy as np
import pandas as pd
import paddle
import csv

from .registry import METRIC
from .base import BaseMetric
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


def BGM_cal_P_R(y, BGM_output):
    """Calculate precession and recall, only use in BCN_bgm.
    """
    precision, recall = cal_P_R(BGM_output, y)
    return precision, recall


def cal_P_R(anchors, scores, acc_threshold=0.5):
    """Calculate precession and recall by anchors and scores.
    """
    scores = paddle.reshape(scores, [scores.shape[-1]])
    anchors = paddle.reshape(anchors, [anchors.shape[-1]])
    # output = (anchors > acc_threshold).int().cpu()
    output = paddle.cast((anchors > acc_threshold), 'int64').cpu()
    # gt=(scores > acc_threshold).int().cpu()
    gt = paddle.cast((scores > acc_threshold), 'int64').cpu()
    TP = 0.0
    FP = 0.0
    FN = 0.0
    if scores.shape[0] == 0:
        return 0.0, 0.0
    for i in range(scores.shape[0]):
        if output[i] == 1:
            if output[i] == gt[i]:
                TP = TP + 1
            else:
                FP = FP + 1
        else:
            if gt[i] == 1:
                FN = FN + 1
    if (TP + FP) == 0:
        return 0.0, 0.0
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall


@METRIC.register
class BcnBgmMetric(BaseMetric):
    """
    Metrics for bgm model of BCN
    """

    def __init__(self, data_size, batch_size, log_interval=1):
        """
        Init for BCN metrics.
        """
        super().__init__(data_size, batch_size, log_interval)
        self.sum_precision = 0.
        self.sum_recall = 0.
        self.cnt4data = 0

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        batch_precision, batch_recall = BGM_cal_P_R(data['match_score'],
                                                    outputs)
        self.sum_precision += batch_precision
        self.sum_recall += batch_recall
        self.cnt4data += 1
        # f1_score = 2 * (batch_precision * batch_recall) / (batch_precision + batch_recall)
        # if batch_id % self.log_interval == 0:
        #     logger.info("Processing................ batch {}, f1 {}".format(batch_id, f1_score))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        f1_score = 2 * ((self.sum_precision / self.cnt4data) * (self.sum_recall / self.cnt4data)) / \
                                    ((self.sum_precision / self.cnt4data) + (self.sum_recall / self.cnt4data))
        logger.info("Processing................ \t acc:{:.4f}\t recall:{:.4f}\t f1:{:.4f}".format(\
                (self.sum_precision / self.cnt4data), (self.sum_recall / self.cnt4data), f1_score))
        # reset
        self.sum_precision = 0.
        self.sum_recall = 0.
        self.cnt4data = 0

        return f1_score


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    """Get each segment of [label, start_time, end_time].
    """
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    """Calculate edit score.
    """
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


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    """Get labels and calculate edit score.
    """
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    """Calculate f-score.
    """
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(
            p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
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


def create_csv(path):
    """Create csv file.
    """
    dir_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(path, "w+", newline='') as file:
        csv_file = csv.writer(file)
        head = ["Acc", "Edit", "F1@10", "F1@25", "F1@50"]
        csv_file.writerow(head)


def append_csv(path, metric_list):
    """Additional written to csv file.
    """
    with open(path, "a+", newline=''
              ) as file:  # 处理csv读写时不同换行符  linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        datas = [metric_list]
        csv_file.writerows(datas)


@METRIC.register
class BcnModelMetric(BaseMetric):
    """
    For Video Segmentation main model.
    """

    def __init__(self,
                 data_size,
                 batch_size,
                 overlap,
                 actions_map_file_path,
                 log_path,
                 dataset,
                 log_interval=1):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval)
        # actions dict generate
        file_ptr = open(actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])

        if os.path.exists(log_path):
            os.remove(log_path)
        create_csv(log_path)
        self.log_path = log_path

        self.overlap = overlap
        self.overlap_len = len(overlap)

        bg_class = ["action_start", "action_end"]
        if dataset == 'gtea':
            bg_class = ['background']
        if dataset == 'breakfast':
            bg_class = ['SIL']
        self.bg_class = bg_class

        self.total_tp = np.zeros(self.overlap_len)
        self.total_fp = np.zeros(self.overlap_len)
        self.total_fn = np.zeros(self.overlap_len)
        self.total_correct = 0
        self.total_edit = 0
        self.total_frame = 0
        self.total_video = 0

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        groundTruth = data['target_tensor']

        outputs_np = outputs.cpu().detach().numpy()
        gt_np = groundTruth.cpu().detach().numpy()[0, :]

        recognition = []
        for i in range(outputs_np.shape[0]):
            recognition = np.concatenate((recognition, [
                list(self.actions_dict.keys())[list(
                    self.actions_dict.values()).index(outputs_np[i])]
            ]))
        recog_content = list(recognition)

        gt_content = []
        for i in range(gt_np.shape[0]):
            gt_content = np.concatenate((gt_content, [
                list(self.actions_dict.keys())[list(
                    self.actions_dict.values()).index(gt_np[i])]
            ]))
        gt_content = list(gt_content)

        tp, fp, fn = np.zeros(self.overlap_len), np.zeros(
            self.overlap_len), np.zeros(self.overlap_len)

        correct = 0
        total = 0
        edit = 0

        for i in range(len(gt_content)):
            total += 1
            #accumulate
            self.total_frame += 1

            if gt_content[i] == recog_content[i]:
                correct += 1
                #accumulate
                self.total_correct += 1

        edit_num = edit_score(recog_content, gt_content, bg_class=self.bg_class)
        edit += edit_num
        self.total_edit += edit_num

        for s in range(self.overlap_len):
            tp1, fp1, fn1 = f_score(recog_content,
                                    gt_content,
                                    self.overlap[s],
                                    bg_class=self.bg_class)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

            # accumulate
            self.total_tp[s] += tp1
            self.total_fp[s] += fp1
            self.total_fn[s] += fn1

        # accumulate
        self.total_video += 1

        Acc = 100 * float(correct) / total
        Edit = (1.0 * edit) / 1.0
        Fscore = dict()
        for s in range(self.overlap_len):
            precision = tp[s] / float(tp[s] + fp[s])
            recall = tp[s] / float(tp[s] + fn[s])

            f1 = 2.0 * (precision * recall) / (precision + recall)

            f1 = np.nan_to_num(f1) * 100
            Fscore[self.overlap[s]] = f1

        # preds ensemble
        # if batch_id % self.log_interval == 0:
        #     logger.info("batch_id:[{:d}] model performence".format(batch_id))
        #     logger.info("Acc: {:.4f}".format(Acc))
        #     logger.info('Edit: {:.4f}'.format(Edit))
        #     for s in range(len(self.overlap)):
        #         logger.info('F1@{:0.2f}: {:.4f}'.format(
        #             self.overlap[s], Fscore[self.overlap[s]]))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        Acc = 100 * float(self.total_correct) / self.total_frame
        Edit = (1.0 * self.total_edit) / self.total_video
        Fscore = dict()
        for s in range(self.overlap_len):
            precision = self.total_tp[s] / float(self.total_tp[s] +
                                                 self.total_fp[s])
            recall = self.total_tp[s] / float(self.total_tp[s] +
                                              self.total_fn[s])

            f1 = 2.0 * (precision * recall) / (precision + recall)

            f1 = np.nan_to_num(f1) * 100
            Fscore[self.overlap[s]] = f1

        # preds ensemble
        logger.info("dataset model performence:")
        logger.info("Acc: {:.4f}".format(Acc))
        logger.info('Edit: {:.4f}'.format(Edit))
        for s in range(len(self.overlap)):
            logger.info('F1@{:0.2f}: {:.4f}'.format(self.overlap[s],
                                                    Fscore[self.overlap[s]]))

        # clear for next epoch
        self.total_tp = np.zeros(self.overlap_len)
        self.total_fp = np.zeros(self.overlap_len)
        self.total_fn = np.zeros(self.overlap_len)
        self.total_correct = 0
        self.total_edit = 0
        self.total_frame = 0
        self.total_video = 0

        # log metric
        metric_list = [Acc, Edit]
        for s in range(self.overlap_len):
            metric_list.append(Fscore[self.overlap[s]])
        append_csv(self.log_path, metric_list)

        return [Acc, Edit, Fscore]
