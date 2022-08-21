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

import numpy
import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddle.static import Variable

from ..registry import LOSSES
from .base import BaseWeightedLoss
from ..framework.localizers.yowo_utils import build_targets


class FocalLoss(nn.Layer):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.

    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()

        if alpha is None:
            self.alpha = paddle.ones(
                [class_num, 1])
            self.alpha.stop_gradient = False
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = (alpha)
                self.alpha.stop_gradient = False
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.shape[0]
        C = inputs.shape[1]
        P = F.softmax(inputs, axis=1)

        tmp = numpy.zeros((N, C))
        class_mask = paddle.to_tensor(tmp, place=inputs.place)
        class_mask.stop_gradient = False
        ids = paddle.reshape(targets, [-1, 1])
        class_mask = F.one_hot(ids.squeeze(-1), class_mask.shape[1])

        if "Place" not in str(inputs.place) and "Place" not in str(self.alpha.place):
            self.alpha = self.alpha.cuda()

        alpha = self.alpha[paddle.reshape(ids.detach(), [-1])]

        probs = paddle.reshape((P * class_mask).sum(1), [-1, 1])

        log_p = probs.log()

        batch_loss = -alpha * (paddle.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


@LOSSES.register()
class RegionLoss(BaseWeightedLoss):
    # for our model anchors has 10 values and number of anchors is 5
    # parameters: 24, 10 float values, 24, 5
    def __init__(self, num_classes, anchors, num_anchors, object_scale, noobject_scale, class_scale, coord_scale):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = [float(x) for x in anchors]
        self.num_anchors = num_anchors
        self.anchor_step = len(self.anchors) // self.num_anchors  # each anchor has 2 parameters
        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale
        self.focalloss = FocalLoss(class_num=self.num_classes, gamma=2, size_average=False)
        self.thresh = 0.6

    def convert2cpu(self, gpu_matrix):
        # return paddle.to_tensor((gpu_matrix.shape), dtype="float32").copy_(gpu_matrix)
        return gpu_matrix.cpu()

    def forward(self, output, target):
        # output : B*A*(4+1+num_classes)*H*W            8*5*29*24*24
        # B: number of batches
        # A: number of anchors
        # 4: 4 parameters for each bounding box
        # 1: confidence score
        # num_classes
        # H: height of the image (in grids)
        # W: width of the image (in grids)
        # for each grid cell, there are A*(4+1+num_classes) parameters
        nB = output.detach().shape[0]  # batch
        nA = self.num_anchors  # anchor_num
        nC = self.num_classes
        nH = output.detach().shape[2]
        nW = output.detach().shape[3]

        # resize the output (all parameters for each anchor can be reached)
        output = paddle.reshape(output, [nB, nA, (5 + nC), nH, nW])
        # anchor's parameter tx

        x = F.sigmoid(
            paddle.reshape(paddle.index_select(output, paddle.to_tensor([0], dtype='int64').cuda(), axis=2),
                           [nB, nA, nH, nW]))
        x.stop_gradient = False
        # anchor's parameter ty
        y = F.sigmoid(
            paddle.reshape(paddle.index_select(output, paddle.to_tensor([1], dtype='int64').cuda(), axis=2),
                           [nB, nA, nH, nW]))
        y.stop_gradient = False
        # anchor's parameter tw
        w = paddle.reshape(paddle.index_select(output, paddle.to_tensor([2], dtype='int64').cuda(), axis=2),
                           [nB, nA, nH, nW])
        w.stop_gradient = False
        # anchor's parameter th
        h = paddle.reshape(paddle.index_select(output, paddle.to_tensor([3], dtype='int64').cuda(), axis=2),
                           [nB, nA, nH, nW])
        h.stop_gradient = False
        # confidence score for each anchor
        conf = F.sigmoid(
            paddle.reshape(paddle.index_select(output, paddle.to_tensor([4], dtype='int64').cuda(), axis=2),
                           [nB, nA, nH, nW]))
        conf.stop_gradient = False
        # anchor's parameter class label
        cls = paddle.index_select(output, paddle.linspace(5, 5 + nC - 1, nC, 'int64').cuda(), axis=2)
        cls.stop_gradient = False
        # resize the data structure so that for every anchor there is a class label in the last dimension
        cls = paddle.reshape(paddle.transpose(paddle.reshape(cls, [nB * nA, nC, nH * nW]), [0, 2, 1]),
                             [nB * nA * nH * nW, nC])

        # for the prediction of localization of each bounding box, there exist 4 parameters (tx, ty, tw, th)
        # pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
        pred_boxes = paddle.zeros([4, nB * nA * nH * nW], dtype='float32').cuda()
        # tx and ty
        grid_x = paddle.reshape(paddle.tile(paddle.tile(paddle.linspace(0, nW - 1, nW), [nH, 1]), [nB * nA, 1, 1]),
                                [nB * nA * nH * nW]).cuda()
        grid_y = paddle.reshape(paddle.tile(paddle.tile(paddle.linspace(0, nH - 1, nH), [nW, 1]).t(), [nB * nA, 1, 1]),
                                [nB * nA * nH * nW]).cuda()
        # for each anchor there are anchor_step variables (with the structure num_anchor*anchor_step)
        # for each row(anchor), the first variable is anchor's width, second is anchor's height
        # pw and ph
        anchor_w = paddle.index_select(paddle.reshape(paddle.to_tensor(self.anchors), [nA, self.anchor_step]),
                                       paddle.to_tensor([0], dtype='int64'), axis=1).cuda()
        anchor_h = paddle.index_select(paddle.reshape(paddle.to_tensor(self.anchors), [nA, self.anchor_step]),
                                       paddle.to_tensor([1], dtype='int64'), axis=1).cuda()
        # for each pixel (grid) repeat the above process (obtain width and height of each grid)
        anchor_w = paddle.reshape(paddle.tile(paddle.tile(anchor_w, [nB, 1]), [1, 1, nH * nW]), [nB * nA * nH * nW])
        anchor_h = paddle.reshape(paddle.tile(paddle.tile(anchor_h, [nB, 1]), [1, 1, nH * nW]), [nB * nA * nH * nW])
        # prediction of bounding box localization
        # x.data and y.data: top left corner of the anchor
        # grid_x, grid_y: tx and ty predictions made by yowo

        x_data = paddle.reshape(x.detach(), [-1])
        y_data = paddle.reshape(y.detach(), [-1])
        w_data = paddle.reshape(w.detach(), [-1])
        h_data = paddle.reshape(h.detach(), [-1])

        pred_boxes[0] = paddle.cast(x_data, dtype='float32') + paddle.cast(grid_x, dtype='float32')  # bx
        pred_boxes[1] = paddle.cast(y_data, dtype='float32') + paddle.cast(grid_y, dtype='float32')  # by
        pred_boxes[2] = paddle.exp(paddle.cast(w_data, dtype='float32')) * paddle.cast(anchor_w, dtype='float32')  # bw
        pred_boxes[3] = paddle.exp(paddle.cast(h_data, dtype='float32')) * paddle.cast(anchor_h, dtype='float32')  # bh
        # the size -1 is inferred from other dimensions
        # pred_boxes (nB*nA*nH*nW, 4)

        pred_boxes = self.convert2cpu(
            paddle.cast(paddle.reshape(paddle.transpose(pred_boxes, (1, 0)), [-1, 4]), dtype='float32'))

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes,
                                                                                                    target.detach(),
                                                                                                    self.anchors, nA,
                                                                                                    nC, \
                                                                                                    nH, nW,
                                                                                                    self.noobject_scale,
                                                                                                    self.object_scale,
                                                                                                    self.thresh)
        cls_mask = (cls_mask == 1)
        #  keep those with high box confidence scores (greater than 0.25) as our final predictions
        nProposals = int((conf > 0.25).sum().detach().item())

        tx = (tx).cuda()
        tx.stop_gradient = False
        ty = ty.cuda()
        ty.stop_gradient = False
        tw = tw.cuda()
        tw.stop_gradient = False
        th = th.cuda()
        th.stop_gradient = False
        tconf = tconf.cuda()
        tconf.stop_gradient = False

        tcls = paddle.reshape(tcls, [-1]).astype('int64')[paddle.reshape(cls_mask, [-1])].cuda()
        tcls.stop_gradient = False

        coord_mask = coord_mask.cuda()
        coord_mask.stop_gradient = False
        conf_mask = conf_mask.cuda().sqrt()
        coord_mask.stop_gradient = False
        cls_mask = paddle.tile(paddle.reshape(cls_mask, [-1, 1]), [1, nC]).cuda()
        cls_mask.stop_gradient = False

        cls = paddle.reshape(cls[cls_mask], [-1, nC])

        # losses between predictions and targets (ground truth)
        # In total 6 aspects are considered as losses:
        # 4 for bounding box location, 2 for prediction confidence and classification seperately
        L1_loss = nn.SmoothL1Loss(reduction='sum')
        loss_x = self.coord_scale * L1_loss(paddle.cast(x, dtype="float32") * coord_mask, tx * coord_mask) / 2.0
        loss_y = self.coord_scale * L1_loss(paddle.cast(y, dtype="float32") * coord_mask, ty * coord_mask) / 2.0
        loss_w = self.coord_scale * L1_loss(paddle.cast(w * coord_mask, dtype="float32"), tw * coord_mask) / 2.0
        loss_h = self.coord_scale * L1_loss(paddle.cast(h * coord_mask, dtype="float32"), th * coord_mask) / 2.0
        loss_conf = nn.MSELoss(reduction='sum')(paddle.cast(conf, dtype="float32") * conf_mask, tconf * conf_mask) / 2.0

        # try focal loss with gamma = 2
        loss_cls = self.class_scale * self.focalloss(cls, tcls)

        # sum of loss
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        return loss, nCorrect


