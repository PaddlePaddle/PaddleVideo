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
import argparse
import pandas as pd

from .registry import METRIC
from .base import BaseMetric
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


def get_labels_scores_start_end_time(input_np,
                                     frame_wise_labels,
                                     actions_dict,
                                     bg_class=["background", "None"]):
    labels = []
    starts = []
    ends = []
    scores = []

    boundary_score_ptr = 0

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
                score = np.mean(
                        input_np[actions_dict[labels[boundary_score_ptr]], \
                            starts[boundary_score_ptr]:(ends[boundary_score_ptr] + 1)]
                        )
                scores.append(score)
                boundary_score_ptr = boundary_score_ptr + 1
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
        score = np.mean(
                    input_np[actions_dict[labels[boundary_score_ptr]], \
                        starts[boundary_score_ptr]:(ends[boundary_score_ptr] + 1)]
                    )
        scores.append(score)
        boundary_score_ptr = boundary_score_ptr + 1

    return labels, starts, ends, scores


def get_labels_start_end_time(frame_wise_labels,
                              bg_class=["background", "None"]):
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


def edit_score(recognized,
               ground_truth,
               norm=True,
               bg_class=["background", "None"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background", "None"]):
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


def boundary_AR(pred_boundary, gt_boundary, overlap_list, max_proposal):

    p_label, p_start, p_end, p_scores = pred_boundary
    y_label, y_start, y_end, _ = gt_boundary

    # sort proposal
    pred_dict = {
        "label": p_label,
        "start": p_start,
        "end": p_end,
        "scores": p_scores
    }
    pdf = pd.DataFrame(pred_dict)
    pdf = pdf.sort_values(by="scores", ascending=False)
    p_label = list(pdf["label"])
    p_start = list(pdf["start"])
    p_end = list(pdf["end"])
    p_scores = list(pdf["scores"])

    # refine AN
    if len(p_label) < max_proposal and len(p_label) > 0:
        p_label = p_label + [p_label[-1]] * (max_proposal - len(p_label))
        p_start = p_start + [p_start[-1]] * (max_proposal - len(p_start))
        p_start = p_start + p_start[len(p_start) -
                                    (max_proposal - len(p_start)):]
        p_end = p_end + [p_end[-1]] * (max_proposal - len(p_end))
        p_scores = p_scores + [p_scores[-1]] * (max_proposal - len(p_scores))
    elif len(p_label) > max_proposal:
        p_label[max_proposal:] = []
        p_start[max_proposal:] = []
        p_end[max_proposal:] = []
        p_scores[max_proposal:] = []

    t_AR = np.zeros(len(overlap_list))

    for i in range(len(overlap_list)):
        overlap = overlap_list[i]

        tp = 0
        fp = 0
        hits = np.zeros(len(y_label))

        for j in range(len(p_label)):
            intersection = np.minimum(p_end[j], y_end) - np.maximum(
                p_start[j], y_start)
            union = np.maximum(p_end[j], y_end) - np.minimum(
                p_start[j], y_start)
            IoU = (1.0 * intersection / union)
            # Get the best scoring segment
            idx = np.array(IoU).argmax()

            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1
        fn = len(y_label) - sum(hits)

        recall = float(tp) / (float(tp) + float(fn))
        t_AR[i] = recall

    AR = np.mean(t_AR)
    return AR


@METRIC.register
class SegmentationMetric(BaseMetric):
    """
    Test for Video Segmentation based model.
    """

    def __init__(self,
                 data_size,
                 batch_size,
                 overlap,
                 actions_map_file_path,
                 log_interval=1,
                 tolerance=5,
                 boundary_threshold=0.7,
                 max_proposal=100):
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

        # cls score
        self.overlap = overlap
        self.overlap_len = len(overlap)

        self.cls_tp = np.zeros(self.overlap_len)
        self.cls_fp = np.zeros(self.overlap_len)
        self.cls_fn = np.zeros(self.overlap_len)
        self.total_correct = 0
        self.total_edit = 0
        self.total_frame = 0
        self.total_video = 0

        # boundary score
        self.max_proposal = max_proposal
        self.AR_at_AN = [[] for _ in range(max_proposal)]

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        groundTruth = data[1]

        predicted = outputs['predict']
        output_np = outputs['output_np']

        outputs_np = predicted.numpy()
        outputs_arr = output_np.numpy()[0, :]
        gt_np = groundTruth.numpy()[0, :]

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

        pred_boundary = get_labels_scores_start_end_time(
            outputs_arr, recog_content, self.actions_dict)
        gt_boundary = get_labels_scores_start_end_time(
            np.ones(outputs_arr.shape), gt_content, self.actions_dict)

        # cls score
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

        edit_num = edit_score(recog_content, gt_content)
        edit += edit_num
        self.total_edit += edit_num

        for s in range(self.overlap_len):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, self.overlap[s])

            # accumulate
            self.cls_tp[s] += tp1
            self.cls_fp[s] += fp1
            self.cls_fn[s] += fn1

        # accumulate
        self.total_video += 1

        # proposal score
        for AN in range(self.max_proposal):
            AR = boundary_AR(pred_boundary,
                             gt_boundary,
                             self.overlap,
                             max_proposal=(AN + 1))
            self.AR_at_AN[AN].append(AR)

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        # cls metric
        Acc = 100 * float(self.total_correct) / self.total_frame
        Edit = (1.0 * self.total_edit) / self.total_video
        Fscore = dict()
        for s in range(self.overlap_len):
            precision = self.cls_tp[s] / float(self.cls_tp[s] + self.cls_fp[s])
            recall = self.cls_tp[s] / float(self.cls_tp[s] + self.cls_fn[s])

            f1 = 2.0 * (precision * recall) / (precision + recall)

            f1 = np.nan_to_num(f1) * 100
            Fscore[self.overlap[s]] = f1

        # proposal metric
        proposal_AUC = np.array(self.AR_at_AN) * 100
        AUC = np.mean(proposal_AUC)
        AR_at_AN1 = np.mean(proposal_AUC[0, :])
        AR_at_AN5 = np.mean(proposal_AUC[4, :])
        AR_at_AN15 = np.mean(proposal_AUC[14, :])

        # log metric
        log_mertic_info = "dataset model performence: "
        # preds ensemble
        log_mertic_info += "Acc: {:.4f}, ".format(Acc)
        log_mertic_info += 'Edit: {:.4f}, '.format(Edit)
        for s in range(len(self.overlap)):
            log_mertic_info += 'F1@{:0.2f}: {:.4f}, '.format(
                self.overlap[s], Fscore[self.overlap[s]])

        # boundary metric
        log_mertic_info += "Auc: {:.4f}, ".format(AUC)
        log_mertic_info += "AR@AN1: {:.4f}, ".format(AR_at_AN1)
        log_mertic_info += "AR@AN5: {:.4f}, ".format(AR_at_AN5)
        log_mertic_info += "AR@AN15: {:.4f}, ".format(AR_at_AN15)
        logger.info(log_mertic_info)

        # log metric
        metric_dict = dict()
        metric_dict['Acc'] = Acc
        metric_dict['Edit'] = Edit
        for s in range(len(self.overlap)):
            metric_dict['F1@{:0.2f}'.format(
                self.overlap[s])] = Fscore[self.overlap[s]]
        metric_dict['Auc'] = AUC
        metric_dict['AR@AN1'] = AR_at_AN1
        metric_dict['AR@AN5'] = AR_at_AN5
        metric_dict['AR@AN15'] = AR_at_AN15

        # clear for next epoch
        # cls
        self.cls_tp = np.zeros(self.overlap_len)
        self.cls_fp = np.zeros(self.overlap_len)
        self.cls_fn = np.zeros(self.overlap_len)
        self.total_correct = 0
        self.total_edit = 0
        self.total_frame = 0
        self.total_video = 0
        # proposal
        self.AR_at_AN = [[] for _ in range(self.max_proposal)]

        return metric_dict
