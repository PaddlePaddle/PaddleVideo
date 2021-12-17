#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import numpy as np
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator():
    """
    MetricsCalculator
    """
    def __init__(self, name, mode, metrics_args):
        """
        init
        """
        self.name = name
        self.mode = mode  # 'train', 'val', 'test'
        self.acc_dict = {}
        self.top_n_list = metrics_args.MODEL.top_n
        self.num_classes = metrics_args.MODEL.num_classes
        self.reset()

    def reset(self):
        """
        reset
        """
        logger.info('Resetting {} metrics...'.format(self.mode))
        for topk in self.top_n_list:
            self.acc_dict['avg_acc%d' % (topk)] = 0.0
        self.aggr_loss = 0.0
        self.aggr_batch_size = 0

    def finalize_metrics(self):
        """finalize_metrics
        """
        for key, value in self.acc_dict.items():
            self.acc_dict[key] = value / self.aggr_batch_size
        self.aggr_loss = self.aggr_loss / self.aggr_batch_size

    def get_computed_metrics(self):
        """get_computed_metrics
        """
        acc_dict = {}
        for key, value in self.acc_dict.items():
            acc_dict[key] = value / self.aggr_batch_size
        aggr_loss = self.aggr_loss / self.aggr_batch_size

        return acc_dict, aggr_loss

    def accumulate(self, loss, softmax, labels):
        """accumulate
        """
        cur_batch_size = softmax.shape[0]
        # if returned loss is None for e.g. test, just set loss to be 0.
        if loss is None:
            cur_loss = 0.
        else:
            cur_loss = np.mean(np.array(loss))  #
        self.aggr_batch_size += cur_batch_size
        self.aggr_loss += cur_loss * cur_batch_size

        for top_k in self.top_n_list:
            self.acc_dict['avg_acc%d' %
                          (top_k)] += cur_batch_size * compute_topk_accuracy(
                              softmax, labels, top_k=top_k) * 100.
        return

    def finalize_and_log_out(self, info=''):
        """finalize_and_log_out
        """
        metrics_dict, loss = self.get_computed_metrics()
        acc_str = []
        for name, value in metrics_dict.items():
            acc_str.append('{}:{},'.format('%s' % name, '%.2f' % value))
        acc_str = '\t'.join(acc_str)
        logger.info(info +
                    '\tLoss: {},\t{}'.format('%.6f' % loss, '%s' % acc_str))
        return


def compute_topk_correct_hits_multilabel(top_k, preds, labels):
    '''Compute the number of corret hits'''
    batch_size = preds.shape[0]
    top_k_preds = np.zeros((batch_size, 10), dtype=np.float32)
    for i in range(batch_size):
        top_k_preds[i, :] = np.argsort(-preds[i, :])[:10]
    correctness = np.zeros(batch_size, dtype=np.float32)
    for i in range(batch_size):
        correc_sum = 0
        for label_id in range(len(labels[i])):
            label_hit = labels[i][label_id]
            if label_hit == 0 or label_hit < 0.1:
                continue
            if label_id in top_k_preds[i, :top_k].astype(np.int32).tolist():
                # correc_sum += 1
                correc_sum = 1
                break
        correctness[i] = correc_sum
    correct_hits = sum(correctness)
    return correct_hits


def compute_topk_correct_hits(top_k, preds, labels):
    '''Compute the number of corret hits'''
    batch_size = preds.shape[0]

    top_k_preds = np.zeros((batch_size, top_k), dtype=np.float32)
    for i in range(batch_size):
        top_k_preds[i, :] = np.argsort(-preds[i, :])[:top_k]

    correctness = np.zeros(batch_size, dtype=np.int32)
    for i in range(batch_size):
        if labels[i] in top_k_preds[i, :].astype(np.int32).tolist():
            correctness[i] = 1
    correct_hits = sum(correctness)

    return correct_hits


def compute_topk_accuracy(softmax, labels, top_k):
    """compute_topk_accuracy
    """
    computed_metrics = {}
    assert labels.shape[0] == softmax.shape[0], "Batch size mismatch."
    aggr_batch_size = labels.shape[0]
    # aggr_top_k_correct_hits = compute_topk_correct_hits(top_k, softmax, labels)
    aggr_top_k_correct_hits = compute_topk_correct_hits_multilabel(
        top_k, softmax, labels)
    # normalize results
    computed_metrics = \
        float(aggr_top_k_correct_hits) / aggr_batch_size

    return computed_metrics


if __name__ == "__main__":
    pred = np.array([[0.5, 0.2, 0.3, 0, 0]])
    label = np.array([[0.5, 0.5, 0, 0, 0]])
    print('pred:  ', pred)
    print('label:  ', label)
    print('Top 1 hits', compute_topk_correct_hits_multilabel(1, pred, label))
    print('Top 5 hits', compute_topk_correct_hits_multilabel(5, pred, label))
