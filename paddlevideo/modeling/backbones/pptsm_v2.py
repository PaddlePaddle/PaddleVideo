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

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, BatchNorm, Conv2D, Dropout, Linear, BatchNorm2D
from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingNormal
import paddle.nn.functional as F

from ..registry import BACKBONES
from ..weight_init import weight_init_
from ...utils import load_ckpt

# MODEL_URLS = {
#     "PPLCNetV2":
#     "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNetV2_base_ssld_pretrained.pdparams",
# }

MODEL_STAGES_PATTERN = {
    "PPLCNet": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
}

NET_CONFIG = {
    # in_channels, kernel_size, split_pw, use_rep, use_se, use_shortcut
    "stage1": [64, 3, False, False, False, False],
    "stage2": [128, 3, False, False, False, False],
    "stage3": [256, 5, True, True, True, False],
    "stage4": [512, 5, False, True, False, True],
}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class GlobalAttention(nn.Layer):
    """
    Lightweight temporal attention module.
    """

    def __init__(self, num_seg=8):
        super().__init__()
        self.fc = nn.Linear(in_features=num_seg,
                            out_features=num_seg,
                            weight_attr=ParamAttr(learning_rate=5.0,
                                                  regularizer=L2Decay(1e-4)),
                            bias_attr=ParamAttr(learning_rate=10.0,
                                                regularizer=L2Decay(0.0)))
        self.num_seg = num_seg

    def forward(self, x):
        _, C, H, W = x.shape
        x0 = x

        x = x.reshape([-1, self.num_seg, C * H * W])
        x = paddle.mean(x, axis=2)  # efficient way of avg_pool
        x = x.squeeze(axis=-1)
        x = self.fc(x)
        attention = F.sigmoid(x)
        attention = attention.reshape(
            (-1, self.num_seg, 1, 1, 1))  #for broadcast

        x0 = x0.reshape([-1, self.num_seg, C, H, W])
        y = paddle.multiply(x0, attention)
        y = y.reshape_([-1, C, H, W])
        return y


class ConvBNLayer(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 use_act=True):
        super().__init__()
        self.use_act = use_act
        self.conv = Conv2D(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=(kernel_size - 1) // 2,
                           groups=groups,
                           weight_attr=ParamAttr(initializer=KaimingNormal()),
                           bias_attr=False)

        self.bn = BatchNorm2D(out_channels,
                              weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                              bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        if self.use_act:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x


class SEModule(nn.Layer):

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(in_channels=channel,
                            out_channels=channel // reduction,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = Conv2D(in_channels=channel // reduction,
                            out_channels=channel,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.hardsigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class RepDepthwiseSeparable(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dw_size=3,
                 split_pw=False,
                 use_rep=False,
                 use_se=False,
                 use_shortcut=False):
        super().__init__()
        self.is_repped = False

        self.dw_size = dw_size
        self.split_pw = split_pw
        self.use_rep = use_rep
        self.use_se = use_se
        self.use_shortcut = True if use_shortcut and stride == 1 and in_channels == out_channels else False

        if self.use_rep:
            self.dw_conv_list = nn.LayerList()
            for kernel_size in range(self.dw_size, 0, -2):
                if kernel_size == 1 and stride != 1:
                    continue
                dw_conv = ConvBNLayer(in_channels=in_channels,
                                      out_channels=in_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      groups=in_channels,
                                      use_act=False)
                self.dw_conv_list.append(dw_conv)
            self.dw_conv = nn.Conv2D(in_channels=in_channels,
                                     out_channels=in_channels,
                                     kernel_size=dw_size,
                                     stride=stride,
                                     padding=(dw_size - 1) // 2,
                                     groups=in_channels)
        else:
            self.dw_conv = ConvBNLayer(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=dw_size,
                                       stride=stride,
                                       groups=in_channels)

        self.act = nn.ReLU()

        if use_se:
            self.se = SEModule(in_channels)

        if self.split_pw:
            pw_ratio = 0.5
            self.pw_conv_1 = ConvBNLayer(in_channels=in_channels,
                                         kernel_size=1,
                                         out_channels=int(out_channels *
                                                          pw_ratio),
                                         stride=1)
            self.pw_conv_2 = ConvBNLayer(in_channels=int(out_channels *
                                                         pw_ratio),
                                         kernel_size=1,
                                         out_channels=out_channels,
                                         stride=1)
        else:
            self.pw_conv = ConvBNLayer(in_channels=in_channels,
                                       kernel_size=1,
                                       out_channels=out_channels,
                                       stride=1)

    def forward(self, x):
        if self.use_rep:
            input_x = x
            if self.is_repped:
                x = self.act(self.dw_conv(x))
            else:
                y = self.dw_conv_list[0](x)
                for dw_conv in self.dw_conv_list[1:]:
                    y += dw_conv(x)
                x = self.act(y)
        else:
            x = self.dw_conv(x)

        if self.use_se:
            x = self.se(x)
        if self.split_pw:
            x = self.pw_conv_1(x)
            x = self.pw_conv_2(x)
        else:
            x = self.pw_conv(x)
        if self.use_shortcut:
            x = x + input_x
        return x

    def rep(self):
        if self.use_rep:
            self.is_repped = True
            kernel, bias = self._get_equivalent_kernel_bias()
            self.dw_conv.weight.set_value(kernel)
            self.dw_conv.bias.set_value(bias)

    def _get_equivalent_kernel_bias(self):
        kernel_sum = 0
        bias_sum = 0
        for dw_conv in self.dw_conv_list:
            kernel, bias = self._fuse_bn_tensor(dw_conv)
            kernel = self._pad_tensor(kernel, to_size=self.dw_size)
            kernel_sum += kernel
            bias_sum += bias
        return kernel_sum, bias_sum

    def _fuse_bn_tensor(self, branch):
        kernel = branch.conv.weight
        running_mean = branch.bn._mean
        running_var = branch.bn._variance
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

    def _pad_tensor(self, tensor, to_size):
        from_size = tensor.shape[-1]
        if from_size == to_size:
            return tensor
        pad = (to_size - from_size) // 2
        return F.pad(tensor, [pad, pad, pad, pad])


class PPTSM_v2_LCNet(nn.Layer):

    def __init__(self,
                 scale,
                 depths,
                 class_num=400,
                 dropout_prob=0,
                 num_seg=8,
                 use_temporal_att=False,
                 pretrained=None,
                 use_last_conv=True,
                 class_expand=1280):
        super().__init__()
        self.scale = scale
        self.use_last_conv = use_last_conv
        self.class_expand = class_expand
        self.num_seg = num_seg
        self.use_temporal_att = use_temporal_att
        self.pretrained = pretrained

        self.stem = nn.Sequential(*[
            ConvBNLayer(in_channels=3,
                        kernel_size=3,
                        out_channels=make_divisible(32 * scale),
                        stride=2),
            RepDepthwiseSeparable(in_channels=make_divisible(32 * scale),
                                  out_channels=make_divisible(64 * scale),
                                  stride=1,
                                  dw_size=3)
        ])

        # stages
        self.stages = nn.LayerList()
        for depth_idx, k in enumerate(NET_CONFIG):
            in_channels, kernel_size, split_pw, use_rep, use_se, use_shortcut = NET_CONFIG[
                k]
            self.stages.append(
                nn.Sequential(*[
                    RepDepthwiseSeparable(in_channels=make_divisible(
                        (in_channels if i == 0 else in_channels * 2) * scale),
                                          out_channels=make_divisible(
                                              in_channels * 2 * scale),
                                          stride=2 if i == 0 else 1,
                                          dw_size=kernel_size,
                                          split_pw=split_pw,
                                          use_rep=use_rep,
                                          use_se=use_se,
                                          use_shortcut=use_shortcut)
                    for i in range(depths[depth_idx])
                ]))

        self.avg_pool = AdaptiveAvgPool2D(1)

        if self.use_last_conv:
            self.last_conv = Conv2D(in_channels=make_divisible(
                NET_CONFIG["stage4"][0] * 2 * scale),
                                    out_channels=self.class_expand,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias_attr=False)
            self.act = nn.ReLU()
            self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")

        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        in_features = self.class_expand if self.use_last_conv else NET_CONFIG[
            "stage4"][0] * 2 * scale
        self.fc = Linear(in_features, class_num)
        if self.use_temporal_att:
            self.global_attention = GlobalAttention(num_seg=self.num_seg)

    def init_weights(self):
        """Initiate the parameters.
        """
        if isinstance(self.pretrained, str) and self.pretrained.strip() != "":
            load_ckpt(self, self.pretrained)
        elif self.pretrained is None or self.pretrained.strip() == "":
            for layer in self.sublayers():
                if isinstance(layer, nn.Conv2D):
                    weight_init_(layer, 'KaimingNormal')
                elif isinstance(layer, nn.BatchNorm2D):
                    weight_init_(layer, 'Constant', value=1)

    def forward(self, x):
        x = self.stem(x)
        count = 0
        for stage in self.stages:
            # only add temporal attention and tsm in stage3 for efficiency
            if count == 2:
                # add temporal attention
                if self.use_temporal_att:
                    x = self.global_attention(x)
                x = F.temporal_shift(x, self.num_seg, 1.0 / self.num_seg)
            count += 1
            x = stage(x)

        x = self.avg_pool(x)
        if self.use_last_conv:
            x = self.last_conv(x)
            x = self.act(x)
            x = self.dropout(x)

        # Feature aggregation
        x = paddle.reshape(x, [-1, self.num_seg, x.shape[1]])
        x = paddle.mean(x, axis=1)
        x = paddle.reshape(x, shape=[-1, self.class_expand])

        x = self.fc(x)
        return x


@BACKBONES.register()
def PPTSM_v2(pretrained=None, use_ssld=False, **kwargs):
    """
    PP-TSM_v2 model.
    Args:
        pretrained: str, means the path of the pretrained model.
    Returns:
        model: nn.Layer.
    """
    model = PPTSM_v2_LCNet(pretrained=pretrained,
                           scale=1.0,
                           depths=[2, 2, 6, 2],
                           dropout_prob=0.2,
                           **kwargs)
    return model
