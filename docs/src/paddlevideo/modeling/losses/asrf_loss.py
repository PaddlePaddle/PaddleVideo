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
# See the License for the specific language governing permissions and
# limitations under the License.

# https://github.com/yiskw713/asrf/libs/loss_fn/__init__.py

import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import sys
import os

from ..registry import LOSSES


class TMSE(nn.Layer):
    """
    Temporal MSE Loss Function
    Proposed in Y. A. Farha et al. MS-TCN: Multi-Stage Temporal Convolutional Network for ActionSegmentation in CVPR2019
    arXiv: https://arxiv.org/pdf/1903.01945.pdf
    """

    def __init__(self, threshold=4, ignore_index=255):
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, preds, gts):

        total_loss = 0.0
        batch_size = preds.shape[0]
        for pred, gt in zip(preds, gts):
            pred = paddle.gather(pred,
                                 paddle.nonzero(gt != self.ignore_index)[:, 0])

            loss = self.mse(F.log_softmax(pred[:, 1:], axis=1),
                            F.log_softmax(pred[:, :-1], axis=1))

            loss = paddle.clip(loss, min=0, max=self.threshold**2)
            total_loss += paddle.mean(loss)

        return total_loss / batch_size


class GaussianSimilarityTMSE(nn.Layer):
    """
    Temporal MSE Loss Function with Gaussian Similarity Weighting
    """

    def __init__(self, threshold=4, sigma=1.0, ignore_index=255):
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction="none")
        self.sigma = sigma

    def forward(self, preds, gts, sim_index):
        """
        Args:
            preds: the output of model before softmax. (N, C, T)
            gts: Ground Truth. (N, T)
            sim_index: similarity index. (N, C, T)
        Return:
            the value of Temporal MSE weighted by Gaussian Similarity.
        """
        total_loss = 0.0
        batch_size = preds.shape[0]
        for pred, gt, sim in zip(preds, gts, sim_index):
            pred = paddle.gather(pred,
                                 paddle.nonzero(gt != self.ignore_index)[:, 0],
                                 axis=1)
            sim = paddle.gather(sim,
                                paddle.nonzero(gt != self.ignore_index)[:, 0],
                                axis=1)

            # calculate gaussian similarity
            diff = sim[:, 1:] - sim[:, :-1]
            similarity = paddle.exp(
                (-1 * paddle.norm(diff, axis=0)) / (2 * self.sigma**2))

            # calculate temporal mse
            loss = self.mse(F.log_softmax(pred[:, 1:], axis=1),
                            F.log_softmax(pred[:, :-1], axis=1))
            loss = paddle.clip(loss, min=0, max=self.threshold**2)

            # gaussian similarity weighting
            loss = similarity * loss

            total_loss += paddle.mean(loss)

        return total_loss / batch_size


class FocalLoss(nn.Layer):

    def __init__(self,
                 weight=None,
                 size_average=True,
                 batch_average=True,
                 ignore_index=255,
                 gamma=2.0,
                 alpha=0.25):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.batch_average = batch_average
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_index,
                                             size_average=size_average)

    def forward(self, logit, target):
        n, _, _ = logit.size()

        logpt = -self.criterion(logit, target.long())
        pt = paddle.exp(logpt)

        if self.alpha is not None:
            logpt *= self.alpha

        loss = -((1 - pt)**self.gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


class ActionSegmentationLoss(nn.Layer):
    """
    Loss Function for Action Segmentation
    You can choose the below loss functions and combine them.
        - Cross Entropy Loss (CE)
        - Focal Loss
        - Temporal MSE (TMSE)
        - Gaussian Similarity TMSE (GSTMSE)
    """

    def __init__(self,
                 num_classes,
                 file_path,
                 label_path,
                 ce=True,
                 focal=True,
                 tmse=False,
                 gstmse=False,
                 weight=None,
                 threshold=4.,
                 ignore_index=255,
                 ce_weight=1.0,
                 focal_weight=1.0,
                 tmse_weight=0.15,
                 gstmse_weight=0.15):
        super().__init__()
        self.criterions = []
        self.weights = []

        self.num_classes = num_classes
        self.file_path = file_path
        self.label_path = label_path
        if weight:
            class_weight = self.get_class_weight()
        else:
            class_weight = None

        if ce:
            self.criterions.append(
                nn.CrossEntropyLoss(weight=class_weight,
                                    ignore_index=ignore_index))
            self.weights.append(ce_weight)

        if focal:
            self.criterions.append(FocalLoss(ignore_index=ignore_index))
            self.weights.append(focal_weight)

        if tmse:
            self.criterions.append(
                TMSE(threshold=threshold, ignore_index=ignore_index))
            self.weights.append(tmse_weight)

        if gstmse:
            self.criterions.append(
                GaussianSimilarityTMSE(threshold=threshold,
                                       ignore_index=ignore_index))
            self.weights.append(gstmse_weight)

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)

    def get_class_weight(self):
        """
        Class weight for CrossEntropy
        Class weight is calculated in the way described in:
            D. Eigen and R. Fergus, “Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture,” in ICCV,
            openaccess: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf
        """
        # load file list
        file_ptr = open(self.file_path, 'r')
        info = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        nums = [0 for i in range(self.num_classes)]
        for i in range(len(info)):
            video_name = info[i]
            file_name = video_name.split('.')[0] + ".npy"
            label_file_path = os.path.join(self.label_path, file_name)
            label = np.load(label_file_path).astype(np.int64)
            num, cnt = np.unique(label, return_counts=True)
            for n, c in zip(num, cnt):
                nums[n] += c

        class_num = paddle.to_tensor(nums, dtype="float32")
        total = class_num.sum().item()
        frequency = class_num / total
        median = paddle.median(frequency)
        class_weight = median / frequency
        return class_weight

    def forward(self, preds, gts, sim_index):
        """
        Args:
            preds: paddle.float (N, C, T).
            gts: paddle.int64 (N, T).
            sim_index: paddle.float (N, C', T).
        """
        loss = 0.0
        for criterion, weight in zip(self.criterions, self.weights):
            if isinstance(criterion, GaussianSimilarityTMSE):
                loss += weight * criterion(preds, gts, sim_index)
            elif isinstance(criterion, nn.CrossEntropyLoss):
                preds_t = paddle.transpose(preds, perm=[0, 2, 1])
                loss += weight * criterion(preds_t, gts)
            else:
                loss += weight * criterion(preds, gts)

        return loss


class BoundaryRegressionLoss(nn.Layer):
    """
    Boundary Regression Loss
        bce: Binary Cross Entropy Loss for Boundary Prediction
        mse: Mean Squared Error
    """

    def __init__(self,
                 file_path,
                 label_path,
                 bce=True,
                 focal=False,
                 mse=False,
                 weight=None,
                 pos_weight=None):
        super().__init__()

        self.criterions = []
        self.file_path = file_path
        self.label_path = label_path

        pos_weight = self.get_pos_weight()

        if bce:
            self.criterions.append(
                nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight))

        if focal:
            self.criterions.append(FocalLoss())

        if mse:
            self.criterions.append(nn.MSELoss())

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)

    def get_pos_weight(self, norm=None):
        """
        pos_weight for binary cross entropy with logits loss
        pos_weight is defined as reciprocal of ratio of positive samples in the dataset
        """
        # load file list
        file_ptr = open(self.file_path, 'r')
        info = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        n_classes = 2  # boundary or not
        nums = [0 for i in range(n_classes)]
        for i in range(len(info)):
            video_name = info[i]
            file_name = video_name.split('.')[0] + ".npy"
            label_file_path = os.path.join(self.label_path, file_name)
            label = np.load(label_file_path).astype(np.int64)
            num, cnt = np.unique(label, return_counts=True)
            for n, c in zip(num, cnt):
                nums[n] += c

        pos_ratio = nums[1] / sum(nums)
        pos_weight = 1 / pos_ratio

        if norm is not None:
            pos_weight /= norm

        return paddle.to_tensor(pos_weight, dtype="float32")

    def forward(self, preds, gts):
        """
        Args:
            preds: paddle.float (N, 1, T).
            gts: paddle.float (N, 1, T).
        """
        loss = 0.0
        batch_size = float(preds.shape[0])

        for criterion in self.criterions:
            for pred, gt in zip(preds, gts):
                loss += criterion(pred, gt)

        return loss / batch_size


@LOSSES.register()
class ASRFLoss(nn.Layer):

    def __init__(self,
                 lambda_bound_loss,
                 num_classes,
                 file_path,
                 label_path,
                 boundary_path,
                 ce=True,
                 asl_focal=True,
                 tmse=False,
                 gstmse=False,
                 asl_weight=None,
                 threshold=4.,
                 ignore_index=255,
                 ce_weight=1.0,
                 focal_weight=1.0,
                 tmse_weight=0.15,
                 gstmse_weight=0.15,
                 bce=True,
                 brl_focal=False,
                 mse=False,
                 brl_weight=None):
        super().__init__()
        self.criterion_cls = ActionSegmentationLoss(ce=ce,
                                                    focal=asl_focal,
                                                    tmse=tmse,
                                                    gstmse=gstmse,
                                                    weight=asl_weight,
                                                    threshold=threshold,
                                                    ignore_index=ignore_index,
                                                    ce_weight=ce_weight,
                                                    focal_weight=focal_weight,
                                                    tmse_weight=tmse_weight,
                                                    gstmse_weight=gstmse_weight,
                                                    file_path=file_path,
                                                    label_path=label_path,
                                                    num_classes=num_classes)
        self.criterion_boundary = BoundaryRegressionLoss(
            bce=bce,
            focal=brl_focal,
            mse=mse,
            weight=brl_weight,
            file_path=file_path,
            label_path=boundary_path)
        self.lambda_bound_loss = lambda_bound_loss

    def forward(self, x, output_cls, label, outputs_boundary, boundary):
        loss = 0.0
        if isinstance(output_cls, list):
            n = len(output_cls)
            for out in output_cls:
                loss += self.criterion_cls(out, label, x) / n
        else:
            loss += self.criterion_cls(output_cls, label, x)

        if isinstance(outputs_boundary, list):
            n = len(outputs_boundary)
            for out in outputs_boundary:
                loss += self.lambda_bound_loss * self.criterion_boundary(
                    out, boundary) / n
        else:
            loss += self.lambda_bound_loss * self.criterion_boundary(
                outputs_boundary, boundary)

        return loss
