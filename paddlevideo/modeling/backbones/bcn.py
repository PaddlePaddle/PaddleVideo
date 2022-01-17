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

import math
import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import copy
from ..registry import BACKBONES


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = len(tensor.shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed \
        for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    else:
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def calculate_gain(nonlinearity=None, a=None):
    """calculate_gain like torch
    """
    if nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if a is not None:
            return math.sqrt(2.0 / (1 + a**2))
        else:
            return math.sqrt(2.0 / (1 + 0.01**2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        return 1


def KaimingUniform_like_torch(weight_npy,
                              mode='fan_in',
                              nonlinearity='leaky_relu'):
    """KaimingUniform_like_torch
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(weight_npy)
    if mode == 'fan_in':
        fan_mode = fan_in
    else:
        fan_mode = fan_out
    a = math.sqrt(5.0)
    gain = calculate_gain(nonlinearity=nonlinearity, a=a)
    std = gain / math.sqrt(fan_mode)
    bound = math.sqrt(3.0) * std
    return np.random.uniform(-bound, bound, weight_npy.shape)


def init_bias(weight_npy, bias_npy):
    """init_bias like torhc
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(weight_npy)
    bound = 1.0 / math.sqrt(fan_in)
    return np.random.uniform(-bound, bound, bias_npy.shape)


class BgmDilatedResidualLayer(nn.Layer):
    """mstcn layer
    """

    def __init__(self, dilation, in_channels, out_channels):
        super(BgmDilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1D(in_channels,
                                      out_channels,
                                      3,
                                      padding=dilation,
                                      dilation=dilation)
        self.conv_1x1 = nn.Conv1D(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        """mstcn layer forward
        """
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)


class FullBGM(nn.Layer):
    """FullBGM in BCN_bgm
    """

    def __init__(self):
        super(FullBGM, self).__init__()
        self.feat_dim = 2048
        self.batch_size = 1
        self.c_hidden = 256
        self.bgm_best_loss = 10000000
        self.bgm_best_f1 = -10000000
        self.bgm_best_precision = -10000000
        self.output_dim = 1
        self.num_layers = 3
        self.conv_in = nn.Conv1D(self.feat_dim, self.c_hidden, 1)
        self.layers = nn.LayerList(
                [copy.deepcopy(BgmDilatedResidualLayer(2 ** (i + 2), self.c_hidden, self.c_hidden)) \
                    for i in range(self.num_layers)]
            )
        self.conv_out = nn.Conv1D(self.c_hidden, self.output_dim, 1)

    def forward(self, x):
        """FullBGM forward
        """
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        out = F.sigmoid(0.01 * out)
        return out

    def init_weights(self):
        """init_weights by kaiming uniform
        """
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv1D):
                layer.weight.set_value(
                    KaimingUniform_like_torch(layer.weight).astype('float32'))
                if layer.bias is not None:
                    layer.bias.set_value(
                        init_bias(layer.weight, layer.bias).astype('float32'))


class ResizedBGM(nn.Layer):
    """ResizedBGM in BCN_bgm
    """

    def __init__(self, dataset):
        super(ResizedBGM, self).__init__()
        self.feat_dim = 2048
        if dataset == 'breakfast' or dataset == 'gtea':
            self.temporal_dim = 300
        elif dataset == '50salads':
            self.temporal_dim = 400
        self.batch_size = 40
        self.batch_size_test = 10
        self.c_hidden = 512
        self.bgm_best_loss = 10000000
        self.bgm_best_f1 = -10000000
        self.output_dim = 1
        self.conv1 = nn.Conv1D(in_channels=self.feat_dim,
                               out_channels=self.c_hidden,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=1)
        self.conv2 = nn.Conv1D(in_channels=self.c_hidden,
                               out_channels=self.c_hidden,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=1)
        self.conv3 = nn.Conv1D(in_channels=self.c_hidden,
                               out_channels=self.output_dim,
                               kernel_size=1,
                               stride=1,
                               padding=0)

    def forward(self, x):
        """ResizedBGM forward
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.sigmoid(self.conv3(x))
        return x

    def init_weights(self):
        """init_weights by kaiming uniform
        """
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv1D):
                layer.weight.set_value(
                    KaimingUniform_like_torch(layer.weight).astype('float32'))
                if layer.bias is not None:
                    layer.bias.set_value(
                        init_bias(layer.weight, layer.bias).astype('float32'))


@BACKBONES.register()
class BcnBgm(nn.Layer):
    """for BCN_bgm
    """

    def __init__(self, dataset, use_full):
        super(BcnBgm, self).__init__()
        if use_full:
            self.bgm = FullBGM()
        else:
            self.bgm = ResizedBGM(dataset)

    def init_weights(self):
        """init_weights by kaiming uniform
        """
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv1D):
                layer.weight.set_value(
                    KaimingUniform_like_torch(layer.weight).astype('float32'))
                if layer.bias is not None:
                    layer.bias.set_value(
                        init_bias(layer.weight, layer.bias).astype('float32'))

    def forward(self, x):
        """bgm forward
        """
        return self.bgm(x)


class SingleStageModel(nn.Layer):
    """SingleStageModel in mstcn
    """

    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1D(dim, num_f_maps, 1)
        self.layers = nn.LayerList([
            copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps))
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1D(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        """forward
        """
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, mask)
        out = self.conv_out(feature) * mask[:, 0:1, :]
        return out, feature * mask[:, 0:1, :]


class DilatedResidualLayer(nn.Layer):
    """DilatedResidualLayer in mstcn
    """

    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1D(in_channels,
                                      out_channels,
                                      3,
                                      padding=dilation,
                                      dilation=dilation)
        self.conv_1x1 = nn.Conv1D(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()  # default value is 0.5
        self.bn = nn.BatchNorm1D(in_channels,
                                 epsilon=1e-08,
                                 momentum=0.1,
                                 use_global_stats=True)

    def forward(self, x, mask, use_bn=False):
        """forward
        """
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        if use_bn:
            out = self.bn(out)
        else:
            out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


def MultiplyList(myList):
    """multiplyList
    """
    result = 1
    for x in myList:
        result = result * x
    return [result]


@BACKBONES.register()
class BcnModel(nn.Layer):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, dataset, use_lbp, num_soft_lbp, \
        pretrained=None):
        super(BcnModel, self).__init__()
        self.num_stages = num_stages  # number of cascade stages
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim,
                                       num_classes)  # cascade stage 1
        stages = [
            copy.deepcopy(
                SingleStageModel(num_layers, num_f_maps,
                                 dim + (s + 1) * num_f_maps, num_classes))
            for s in range(num_stages - 1)
        ]
        self.stages = nn.LayerList(stages)  # cascade stage 2,...,n
        self.stageF = SingleStageModel(num_layers, 64, num_classes,
                                       num_classes)  # fusion stage
        self.bgm = FullBGM()
        self.lbp_in = LocalBarrierPooling(7, alpha=1)
        self.use_lbp = use_lbp
        self.num_soft_lbp = num_soft_lbp
        self.num_classes = num_classes
        if dataset == '50salads':
            self.lbp_out = LocalBarrierPooling(99, alpha=0.2)  # has lbp_post
        if dataset == 'breakfast':
            self.lbp_out = LocalBarrierPooling(159, alpha=0.3)  # has lbp_post
        if dataset == 'gtea':
            self.lbp_out = LocalBarrierPooling(
                99, alpha=1
            )  # no lbp_post for gtea (because of bad barrier quality of resized BGM due to small dataset size), so alpha=1

    def init_weights(self):
        """init_weights by kaiming uniform
        """
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv1D):
                layer.weight.set_value(
                    KaimingUniform_like_torch(layer.weight).astype('float32'))
                if layer.bias is not None:
                    layer.bias.set_value(
                        init_bias(layer.weight, layer.bias).astype('float32'))

    def forward(self, x, mask, gt_target=None, soft_threshold=0.8):
        """ forward"""
        mask.stop_gradient = True
        x.stop_gradient = True
        adjusted_weight = mask[:, 0:1, :].clone().detach().unsqueeze(
            0)  # weights for SC
        for i in range(self.num_stages - 1):
            adjusted_weight = paddle.concat(
                (adjusted_weight, mask[:,
                                       0:1, :].clone().detach().unsqueeze(0)))
        confidence = []
        feature = []
        if gt_target is not None:
            gt_target = gt_target.unsqueeze(0)

        # stage 1
        out1, feature1 = self.stage1(x, mask)
        outputs = out1.unsqueeze(0)
        feature.append(feature1)
        confidence.append(F.softmax(out1, axis=1) * mask[:, 0:1, :])
        confidence[0].stop_gradient = True

        if gt_target is None:
            max_conf = paddle.max(confidence[0], axis=1)
            max_conf = max_conf.unsqueeze(1).clone().detach()
            max_conf.stop_gradient = True
            decrease_flag = (max_conf > soft_threshold)
            decrease_flag = paddle.cast(decrease_flag, 'float32')
            increase_flag = mask[:, 0:1, :].clone().detach() - decrease_flag
            adjusted_weight[1] = max_conf.neg().exp(
            ) * decrease_flag + max_conf.exp() * increase_flag  # for stage 2
        else:
            one_hot = F.one_hot(gt_target[0], self.num_classes)
            gt_conf = ((confidence[0] *
                        paddle.transpose(one_hot, [0, 2, 1])).sum(1))[0]
            gt_conf = paddle.to_tensor(gt_conf).unsqueeze(0).unsqueeze(0)
            decrease_flag = (gt_conf > soft_threshold)
            decrease_flag = paddle.cast(decrease_flag, 'float32')
            increase_flag = mask[:, 0:1, :].clone().detach() - decrease_flag
            adjusted_weight[1] = gt_conf.neg().exp(
            ) * decrease_flag + gt_conf.exp() * increase_flag

        # stage 2,...,n
        curr_stage = 0
        for s in self.stages:
            # for s_i in range(self.num_stages - 2):
            curr_stage = curr_stage + 1
            temp = feature[0]
            for i in range(1, len(feature)):
                temp = paddle.concat(
                    (temp, feature[i]), axis=1) * mask[:, 0:1, :]
            temp = paddle.concat((temp, x), axis=1)
            curr_out, curr_feature = s(temp, mask)
            outputs = paddle.concat((outputs, curr_out.unsqueeze(0)), axis=0)
            feature.append(curr_feature)
            confidence.append(F.softmax(curr_out, axis=1) * mask[:, 0:1, :])
            confidence[curr_stage].stop_gradient = True
            if curr_stage < self.num_stages - 1:  # curr_stage starts from 0

                if gt_target is None:
                    max_conf = paddle.max(confidence[curr_stage], axis=1)
                    max_conf = max_conf.unsqueeze(1).clone().detach()
                    max_conf.stop_gradient = True
                    decrease_flag = (max_conf > soft_threshold)
                    decrease_flag = paddle.cast(decrease_flag, 'float32')
                    increase_flag = mask[:, 0:1, :].clone().detach(
                    ) - decrease_flag
                    adjusted_weight[curr_stage + 1] = max_conf.neg().exp(
                    ) * decrease_flag + max_conf.exp(
                    ) * increase_flag  # output the weight for the next stage
                else:
                    one_hot = F.one_hot(gt_target[0], self.num_classes)
                    gt_conf = ((confidence[curr_stage] *
                                paddle.transpose(one_hot, [0, 2, 1])).sum(1))[0]
                    gt_conf = paddle.to_tensor(gt_conf).unsqueeze(0).unsqueeze(
                        0)
                    decrease_flag = (gt_conf > soft_threshold)
                    decrease_flag = paddle.cast(decrease_flag, 'float32')
                    increase_flag = mask[:, 0:1, :].clone().detach(
                    ) - decrease_flag
                    adjusted_weight[curr_stage + 1] = gt_conf.neg().exp(
                    ) * decrease_flag + gt_conf.exp() * increase_flag

        output_weight = adjusted_weight.detach()
        output_weight.stop_gradient = True
        adjusted_weight = adjusted_weight / paddle.sum(
            adjusted_weight, 0)  # normalization among stages
        temp = F.softmax(out1, axis=1) * adjusted_weight[0]
        for i in range(1, self.num_stages):
            temp += F.softmax(outputs[i], axis=1) * adjusted_weight[i]
        confidenceF = temp * mask[:, 0:1, :]  # input of fusion stage

        #  Inner LBP for confidenceF
        barrier, BGM_output = self.fullBarrier(x)
        if self.use_lbp:
            confidenceF = self.lbp_in(confidenceF, barrier)

        #  fusion stage: for more consistent output because of the combination of cascade stages may have much fluctuations
        out, _ = self.stageF(confidenceF, mask)  # use mixture of cascade stages

        #  Final LBP for output
        if self.use_lbp:
            for i in range(self.num_soft_lbp):
                out = self.lbp_out(out, barrier)

        confidence_last = paddle.clip(
            F.softmax(out, axis=1), min=1e-4, max=1 -
            1e-4) * mask[:, 0:1, :]  # torch.clamp for training stability
        outputs = paddle.concat((outputs, confidence_last.unsqueeze(0)), axis=0)
        return outputs, BGM_output, output_weight

    def fullBarrier(self, feature_tensor):
        """fullBarrier
        """
        BGM_output = self.bgm(feature_tensor)
        barrier = BGM_output
        return barrier, BGM_output


def im2col(input_data, kh, kw, stride=1, pad=0, dilation=1):
    """
    calculate im2col
    """
    N, C, H, W = input_data.shape
    dh, dw = dilation * (kh - 1) + 1, dilation * (kw - 1) + 1
    h_out = (H + 2 * pad - dh) // stride + 1
    w_out = (W + 2 * pad - dw) // stride + 1
    img = F.pad(input_data, [pad, pad, pad, pad], "constant", value=0)
    col = paddle.zeros((N, C, dh, dw, h_out, w_out))

    for y in range(dh):
        y_max = y + stride * h_out
        for x in range(dw):
            x_max = x + stride * w_out
            col[:, :, y, x, :, :] += img[:, :, y:y_max:stride, x:x_max:stride]
    res = col.reshape([N, C * dh * dw, h_out * w_out])
    return res


def unfold_1d(x, kernel_size=7, pad_value=0):
    """unfold_1d
    """
    B, C, T = x.shape
    padding = kernel_size // 2
    x = x.unsqueeze(-1)
    x = F.pad(x, (0, 0, padding, padding), value=pad_value)
    x = paddle.cast(x, 'float32')
    D = F.unfold(x, [kernel_size, 1])
    # D = im2col(x, kernel_size, 1)
    return paddle.reshape(D, [B, C, kernel_size, T])


def dual_barrier_weight(b, kernel_size=7, alpha=0.2):
    """dual_barrier_weight
    """
    K = kernel_size
    b = unfold_1d(b, kernel_size=K, pad_value=20)
    # b: (B, 1, K, T)
    HL = K // 2
    left = paddle.flip(
        paddle.cumsum(paddle.flip(b[:, :, :HL + 1, :], [2]), axis=2),
        [2])[:, :, :-1, :]
    right = paddle.cumsum(b[:, :, -HL - 1:, :], axis=2)[:, :, 1:, :]
    middle = paddle.zeros_like(b[:, :, 0:1, :])
    #middle = b[:, :, HL:-HL, :]
    weight = alpha * paddle.concat((left, middle, right), axis=2)
    return weight.neg().exp()


class LocalBarrierPooling(nn.Layer):
    """LBP in BCN paper
    """

    def __init__(self, kernel_size=99, alpha=0.2):
        super(LocalBarrierPooling, self).__init__()
        self.kernel_size = kernel_size
        self.alpha = alpha

    def forward(self, x, barrier):
        """
        x: (B, C, T)
        barrier: (B, 1, T) (>=0)
        """
        xs = unfold_1d(x, self.kernel_size)
        w = dual_barrier_weight(barrier, self.kernel_size, self.alpha)
        return (xs * w).sum(axis=2) / ((w).sum(axis=2) + np.exp(-10))
