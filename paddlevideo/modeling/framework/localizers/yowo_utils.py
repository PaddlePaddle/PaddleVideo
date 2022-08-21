# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import math
import paddle
import paddle.nn as nn
import numpy as np
from builtins import range as xrange


def truths_length(truths):
    for i in range(50):
        if truths[i][1] == 0:
            return i


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = paddle.zeros([len(boxes)])
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]

    sortIds = paddle.argsort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    box_j[4] = 0
    return out_boxes


def convert2cpu(gpu_matrix):
    float_32_g = gpu_matrix.astype('float32')
    return float_32_g.cpu()


def convert2cpu_long(gpu_matrix):
    int_64_g = gpu_matrix.astype('int64')
    return int_64_g.cpu()


def get_region_boxes(output, conf_thresh=0.005, num_classes=24,
                     anchors=[0.70458, 1.18803, 1.26654, 2.55121, 1.59382,
                              4.08321, 2.30548, 4.94180, 3.52332, 5.91979],
                     num_anchors=5, only_objectness=1, validation=False):
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.shape[0]
    assert (output.shape[1] == (5 + num_classes) * num_anchors)
    h = output.shape[2]
    w = output.shape[3]
    all_boxes = []
    output = paddle.reshape(
        output, [batch * num_anchors, 5 + num_classes, h * w])
    output = paddle.transpose(output, (1, 0, 2))
    output = paddle.reshape(
        output, [5 + num_classes, batch * num_anchors * h * w])

    grid_x = paddle.linspace(0, w - 1, w)
    grid_x = paddle.tile(grid_x, [h, 1])
    grid_x = paddle.tile(grid_x, [batch * num_anchors, 1, 1])
    grid_x = paddle.reshape(grid_x, [batch * num_anchors * h * w]).cuda()

    grid_y = paddle.linspace(0, h - 1, h)
    grid_y = paddle.tile(grid_y, [w, 1]).t()
    grid_y = paddle.tile(grid_y, [batch * num_anchors, 1, 1])
    grid_y = paddle.reshape(grid_y, [batch * num_anchors * h * w]).cuda()

    sigmoid = nn.Sigmoid()
    xs = sigmoid(output[0]) + grid_x
    ys = sigmoid(output[1]) + grid_y

    anchor_w = paddle.to_tensor(anchors)
    anchor_w = paddle.reshape(anchor_w, [num_anchors, anchor_step])
    anchor_w = paddle.index_select(anchor_w, index=paddle.to_tensor(
        np.array([0]).astype('int32')), axis=1)

    anchor_h = paddle.to_tensor(anchors)
    anchor_h = paddle.reshape(anchor_h, [num_anchors, anchor_step])
    anchor_h = paddle.index_select(anchor_h, index=paddle.to_tensor(
        np.array([1]).astype('int32')), axis=1)

    anchor_w = paddle.tile(anchor_w, [batch, 1])
    anchor_w = paddle.tile(anchor_w, [1, 1, h * w])
    anchor_w = paddle.reshape(anchor_w, [batch * num_anchors * h * w]).cuda()

    anchor_h = paddle.tile(anchor_h, [batch, 1])
    anchor_h = paddle.tile(anchor_h, [1, 1, h * w])
    anchor_h = paddle.reshape(anchor_h, [batch * num_anchors * h * w]).cuda()

    ws = paddle.exp(output[2]) * anchor_w
    hs = paddle.exp(output[3]) * anchor_h

    det_confs = sigmoid(output[4])

    cls_confs = paddle.to_tensor(output[5:5 + num_classes], stop_gradient=True)
    cls_confs = paddle.transpose(cls_confs, [1, 0])
    s = nn.Softmax()
    cls_confs = paddle.to_tensor(s(cls_confs))

    cls_max_confs = paddle.max(cls_confs, axis=1)
    cls_max_ids = paddle.argmax(cls_confs, axis=1)

    cls_max_confs = paddle.reshape(cls_max_confs, [-1])
    cls_max_ids = paddle.reshape(cls_max_ids, [-1])

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors

    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.reshape([-1, num_classes]))
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h,
                               det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    return all_boxes


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(float(box1[0] - box1[2] / 2.0),
                 float(box2[0] - box2[2] / 2.0))
        Mx = max(float(box1[0] + box1[2] / 2.0),
                 float(box2[0] + box2[2] / 2.0))
        my = min(float(box1[1] - box1[3] / 2.0),
                 float(box2[1] - box2[3] / 2.0))
        My = max(float(box1[1] + box1[3] / 2.0),
                 float(box2[1] + box2[3] / 2.0))
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return paddle.to_tensor(0.0)

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = paddle.min(boxes1[0], boxes2[0])
        Mx = paddle.max(boxes1[2], boxes2[2])
        my = paddle.min(boxes1[1], boxes2[1])
        My = paddle.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = paddle.min(paddle.stack(
            [boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0], axis=0), axis=0)
        Mx = paddle.max(paddle.stack(
            [boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0], axis=0), axis=0)
        my = paddle.min(paddle.stack(
            [boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0], axis=0), axis=0)
        My = paddle.max(paddle.stack(
            [boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0], axis=0), axis=0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = paddle.cast(cw <= 0, dtype="int32") + \
        paddle.cast(ch <= 0, dtype="int32") > 0
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


# this function works for building the groud truth
def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale,
                  sil_thresh):
    # nH, nW here are number of grids in y and x directions (7, 7 here)
    nB = target.shape[0]  # batch size
    nA = num_anchors  # 5 for our case
    nC = num_classes
    anchor_step = len(anchors) // num_anchors
    conf_mask = paddle.ones([nB, nA, nH, nW]) * noobject_scale
    coord_mask = paddle.zeros([nB, nA, nH, nW])
    cls_mask = paddle.zeros([nB, nA, nH, nW])
    tx = paddle.zeros([nB, nA, nH, nW])
    ty = paddle.zeros([nB, nA, nH, nW])
    tw = paddle.zeros([nB, nA, nH, nW])
    th = paddle.zeros([nB, nA, nH, nW])
    tconf = paddle.zeros([nB, nA, nH, nW])
    tcls = paddle.zeros([nB, nA, nH, nW])

    # for each grid there are nA anchors
    # nAnchors is the number of anchor for one image
    nAnchors = nA * nH * nW
    nPixels = nH * nW
    # for each image
    for b in xrange(nB):
        # get all anchor boxes in one image
        # (4 * nAnchors)
        cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
        # initialize iou score for each anchor
        cur_ious = paddle.zeros([nAnchors])
        for t in xrange(50):
            # for each anchor 4 coordinate parameters, already in the coordinate system for the whole image
            # this loop is for anchors in each image
            # for each anchor 5 parameters are available (class, x, y, w, h)
            if target[b][t * 5 + 1] == 0:
                break
            gx = target[b][t * 5 + 1] * nW
            gy = target[b][t * 5 + 2] * nH
            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            # groud truth boxes
            cur_gt_boxes = paddle.tile(paddle.to_tensor(
                [gx, gy, gw, gh], dtype='float32').t(), [nAnchors, 1]).t()
            # bbox_ious is the iou value between orediction and groud truth
            cur_ious = paddle.max(
                paddle.stack([cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False)], axis=0), axis=0)
        # if iou > a given threshold, it is seen as it includes an object
        # conf_mask[b][cur_ious>sil_thresh] = 0
        conf_mask_t = paddle.reshape(conf_mask, [nB, -1])
        conf_mask_t[b, cur_ious > sil_thresh] = 0
        conf_mask_tt = paddle.reshape(conf_mask_t[b], [nA, nH, nW])
        conf_mask[b] = conf_mask_tt

    # number of ground truth
    nGT = 0
    nCorrect = 0
    for b in xrange(nB):
        # anchors for one batch (at least batch size, and for some specific classes, there might exist more than one anchor)
        for t in xrange(50):
            if target[b][t * 5 + 1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            # the values saved in target is ratios
            # times by the width and height of the output feature maps nW and nH
            gx = target[b][t * 5 + 1] * nW
            gy = target[b][t * 5 + 2] * nH
            gi = int(gx)
            gj = int(gy)

            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            gt_box = [0, 0, gw, gh]
            for n in xrange(nA):
                # get anchor parameters (2 values)
                aw = anchors[anchor_step * n]
                ah = anchors[anchor_step * n + 1]
                anchor_box = [0, 0, aw, ah]
                # only consider the size (width and height) of the anchor box
                iou = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                # get the best anchor form with the highest iou
                if iou > best_iou:
                    best_iou = iou
                    best_n = n

            # then we determine the parameters for an anchor (4 values together)
            gt_box = [gx, gy, gw, gh]
            # find corresponding prediction box
            pred_box = pred_boxes[b * nAnchors +
                                  best_n * nPixels + gj * nW + gi]

            # only consider the best anchor box, for each image
            coord_mask[b, best_n, gj, gi] = 1
            cls_mask[b, best_n, gj, gi] = 1

            # in this cell of the output feature map, there exists an object
            conf_mask[b, best_n, gj, gi] = object_scale
            tx[b, best_n, gj, gi] = paddle.cast(
                target[b][t * 5 + 1] * nW - gi, dtype='float32')
            ty[b, best_n, gj, gi] = paddle.cast(
                target[b][t * 5 + 2] * nH - gj, dtype='float32')
            tw[b, best_n, gj, gi] = math.log(
                gw / anchors[anchor_step * best_n])
            th[b, best_n, gj, gi] = math.log(
                gh / anchors[anchor_step * best_n + 1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)  # best_iou
            # confidence equals to iou of the corresponding anchor
            tconf[b, best_n, gj, gi] = paddle.cast(iou, dtype='float32')
            tcls[b, best_n, gj, gi] = paddle.cast(
                target[b][t * 5], dtype='float32')
            # if ious larger than 0.5, we justify it as a correct prediction
            if iou > 0.5:
                nCorrect = nCorrect + 1
    # true values are returned
    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls
