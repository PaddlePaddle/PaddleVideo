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
from paddle import ParamAttr
from ..registry import BACKBONES
import paddle.nn.functional as F

def _get_interp1d_bin_mask(seg_xmin, seg_xmax, tscale, num_sample,
                           num_sample_perbin):
    """ generate sample mask for a boundary-matching pair """
    plen = float(seg_xmax - seg_xmin)
    plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
    total_samples = [
        seg_xmin + plen_sample * ii
        for ii in range(num_sample * num_sample_perbin)
    ]
    p_mask = []
    for idx in range(num_sample):
        bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) *
                                    num_sample_perbin]
        bin_vector = np.zeros([tscale])
        for sample in bin_samples:
            sample_upper = math.ceil(sample)
            sample_decimal, sample_down = math.modf(sample)
            if (tscale - 1) >= int(sample_down) >= 0:
                bin_vector[int(sample_down)] += 1 - sample_decimal
            if (tscale - 1) >= int(sample_upper) >= 0:
                bin_vector[int(sample_upper)] += sample_decimal
        bin_vector = 1.0 / num_sample_perbin * bin_vector
        p_mask.append(bin_vector)
    p_mask = np.stack(p_mask, axis=1)
    return p_mask


def get_interp1d_mask(tscale, dscale, prop_boundary_ratio, num_sample,
                      num_sample_perbin):
    """ generate sample mask for each point in Boundary-Matching Map """
    mask_mat = []
    for start_index in range(tscale):
        mask_mat_vector = []
        for duration_index in range(dscale):
            if start_index + duration_index < tscale:
                p_xmin = start_index
                p_xmax = start_index + duration_index
                center_len = float(p_xmax - p_xmin) + 1
                sample_xmin = p_xmin - center_len * prop_boundary_ratio
                sample_xmax = p_xmax + center_len * prop_boundary_ratio
                p_mask = _get_interp1d_bin_mask(sample_xmin, sample_xmax,
                                                tscale, num_sample,
                                                num_sample_perbin)
            else:
                p_mask = np.zeros([tscale, num_sample])
            mask_mat_vector.append(p_mask)
        mask_mat_vector = np.stack(mask_mat_vector, axis=2)
        mask_mat.append(mask_mat_vector)
    mask_mat = np.stack(mask_mat, axis=3)
    mask_mat = mask_mat.astype(np.float32)

    sample_mask = np.reshape(mask_mat, [tscale, -1])
    return sample_mask


def init_params(name, in_channels, kernel_size):
    fan_in = in_channels * kernel_size * 1
    k = 1. / math.sqrt(fan_in)
    param_attr = ParamAttr(name=name,
                           initializer=paddle.nn.initializer.Uniform(low=-k,
                                                                     high=k))
    return param_attr


class LocalGlobalTemporalEncoder1(paddle.nn.Layer):

    def __init__(
        self, 
        input_dim=256, 
        dropout=0.0, 
        temporal_scale=300, 
        window_size=9
        ):

        super(LocalGlobalTemporalEncoder1, self).__init__()

        dim_feedforward = 256

        self.self_atten = GlobalLocalAttention(
            input_dim, 
            num_heads=8, 
            dropout=0.0, 
            temporal_scale=temporal_scale, 
            window_size=window_size)

        self.linear1 = paddle.nn.Linear(
            in_features=input_dim, 
            out_features=dim_feedforward,
            weight_attr=ParamAttr(name="LGTE_L11_w"),
            bias_attr=ParamAttr(name="LGTE_L11_b"))

        self.dropout = paddle.nn.Dropout(dropout)

        self.linear2 = paddle.nn.Linear(
            in_features=dim_feedforward, 
            out_features=input_dim,
            weight_attr=ParamAttr(name="LGTE_L12_w"),
            bias_attr=ParamAttr(name="LGTE_L12_b"))

        self.norm1 = paddle.nn.LayerNorm(input_dim)

        self.norm2 = paddle.nn.LayerNorm(input_dim)

        self.dropout1 = paddle.nn.Dropout(dropout)

        self.dropout2 = paddle.nn.Dropout(dropout)

    def forward(self, features):# 2 256 300
        
        src = features.transpose([0, 2, 1])# 2 300 256
        q = k = src
        src2 = self.self_atten(q, k, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose([0, 2, 1])
        return src


class LocalGlobalTemporalEncoder2(paddle.nn.Layer):

    def __init__(
        self, 
        input_dim=256, 
        dropout=0.0, 
        temporal_scale=300, 
        window_size=9
        ):

        super(LocalGlobalTemporalEncoder2, self).__init__()

        dim_feedforward = 256

        self.self_atten = GlobalLocalAttention(
            input_dim, 
            num_heads=8, 
            dropout=0.0, 
            temporal_scale=temporal_scale, 
            window_size=window_size)

        self.linear1 = paddle.nn.Linear(
            in_features=input_dim, 
            out_features=dim_feedforward,
            weight_attr=ParamAttr(name="LGTE_L21_w"),
            bias_attr=ParamAttr(name="LGTE_L21_b"))

        self.dropout = paddle.nn.Dropout(dropout)

        self.linear2 = paddle.nn.Linear(
            in_features=dim_feedforward, 
            out_features=input_dim,
            weight_attr=ParamAttr(name="LGTE_L22_w"),
            bias_attr=ParamAttr(name="LGTE_L22_b"))

        self.norm1 = paddle.nn.LayerNorm(input_dim)

        self.norm2 = paddle.nn.LayerNorm(input_dim)

        self.dropout1 = paddle.nn.Dropout(dropout)

        self.dropout2 = paddle.nn.Dropout(dropout)

    def forward(self, features):# 2 256 300
        
        src = features.transpose([0, 2, 1])# 2 300 256
        q = k = src
        src2 = self.self_atten(q, k, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose([0, 2, 1])
        return src

        
class GlobalLocalAttention(paddle.nn.Layer):

    def __init__(
        self, 
        input_dim, 
        num_heads, 
        dropout, 
        temporal_scale=300, 
        window_size=9
        ):
        super(GlobalLocalAttention, self).__init__()
        self.num_heads = num_heads
        self.temporal_scale = temporal_scale
        self.scale_attention = paddle.nn.MultiHeadAttention(input_dim,
                                                            num_heads=self.num_heads,
                                                            dropout=dropout)                                   
        self.mask_matrix = self._mask_matrix(window_size)

    def _mask_matrix(self, window_size):
        m = paddle.zeros((1, self.num_heads,
                        self.temporal_scale,
                        self.temporal_scale), dtype=bool)
        w_len = window_size
        local_len = self.num_heads // 2
        # paddle unwanted: 0
        # pytorch unwanted: 1
        for i in range(local_len):
            for j in range(self.temporal_scale):
                for k in range(w_len):
                    m[0, i, j, min(max(j - w_len // 2 + k, 0), self.temporal_scale - 1)] = True
        for i in range(local_len, self.num_heads):
            m[0, i, :, :] = True

        return m

    def forward(self, query, key, value):
        # print(query.shape)
        b = query.shape[0]
        mask = paddle.tile(self.mask_matrix, repeat_times=[b, 1, 1, 1])
        # print(mask.shape)
        r = self.scale_attention(query, key, value, attn_mask=mask)
        return r


@BACKBONES.register()
class ppTCANet(paddle.nn.Layer):
    """ppTCANet model
    Args:
        tscale (int): sequence length, default 100.
        dscale (int): max duration length, default 100.
        prop_boundary_ratio (float): ratio of expanded temporal region in proposal boundary, default 0.5.
        num_sample (int): number of samples betweent starting boundary and ending boundary of each propoasl, default 32.
        num_sample_perbin (int):  number of selected points in each sample, default 3.
    """

    def __init__(
        self,
        tscale,
        dscale,
        prop_boundary_ratio,
        num_sample,
        num_sample_perbin,
        feat_dim=400,
    ):
        super(TCANetpp, self).__init__()

        #init config
        self.feat_dim = feat_dim
        self.tscale = tscale
        self.dscale = dscale
        self.prop_boundary_ratio = prop_boundary_ratio
        self.num_sample = num_sample
        self.num_sample_perbin = num_sample_perbin

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        # LGTE
        self.lgte1 = LocalGlobalTemporalEncoder1(
            input_dim=self.hidden_dim_1d, 
            temporal_scale=self.tscale)
        
        # SE block
        self.se_avg_pool = paddle.nn.layer.AdaptiveAvgPool1D(output_size=1)			
        self.se_linear_1 = paddle.nn.Linear(
            in_features=256, 
            out_features=16,
            weight_attr=ParamAttr(name="SE_w_1"),
            bias_attr=None)
        self.se_act_1 = paddle.nn.ReLU()
        self.se_linear_2 = paddle.nn.Linear(
            in_features=16, 
            out_features=256,
            weight_attr=ParamAttr(name="SE_w_2"),
            bias_attr=None)
        self.se_act_2 = paddle.nn.Sigmoid()

        # Base Module
        self.b_conv1 = paddle.nn.Conv1D(
            in_channels=self.feat_dim,
            out_channels=self.hidden_dim_1d,
            kernel_size=3,
            padding=1,
            groups=4,
            weight_attr=init_params('Base_1_w', self.feat_dim, 3),
            bias_attr=init_params('Base_1_b', self.feat_dim, 3))
        self.b_conv1_act = paddle.nn.ReLU()

        self.b_conv2 = paddle.nn.Conv1D(
            in_channels=self.hidden_dim_1d,
            out_channels=self.hidden_dim_1d,
            kernel_size=3,
            padding=1,
            groups=4,
            weight_attr=init_params('Base_2_w', self.hidden_dim_1d, 3),
            bias_attr=init_params('Base_2_b', self.hidden_dim_1d, 3))
        self.b_conv2_act = paddle.nn.ReLU()

        # Temporal Evaluation Module
        self.ts_conv1 = paddle.nn.Conv1D(
            in_channels=self.hidden_dim_1d,
            out_channels=self.hidden_dim_1d,
            kernel_size=3,
            padding=1,
            groups=4,
            weight_attr=init_params('TEM_s1_w', self.hidden_dim_1d, 3),
            bias_attr=init_params('TEM_s1_b', self.hidden_dim_1d, 3))
        self.ts_conv1_act = paddle.nn.ReLU()

        self.ts_conv2 = paddle.nn.Conv1D(
            in_channels=self.hidden_dim_1d,
            out_channels=1,
            kernel_size=1,
            padding=0,
            groups=1,
            weight_attr=init_params('TEM_s2_w', self.hidden_dim_1d, 1),
            bias_attr=init_params('TEM_s2_b', self.hidden_dim_1d, 1))
        self.ts_conv2_act = paddle.nn.Sigmoid()

        self.te_conv1 = paddle.nn.Conv1D(
            in_channels=self.hidden_dim_1d,
            out_channels=self.hidden_dim_1d,
            kernel_size=3,
            padding=1,
            groups=4,
            weight_attr=init_params('TEM_e1_w', self.hidden_dim_1d, 3),
            bias_attr=init_params('TEM_e1_b', self.hidden_dim_1d, 3))
        self.te_conv1_act = paddle.nn.ReLU()
        self.te_conv2 = paddle.nn.Conv1D(
            in_channels=self.hidden_dim_1d,
            out_channels=1,
            kernel_size=1,
            padding=0,
            groups=1,
            weight_attr=init_params('TEM_e2_w', self.hidden_dim_1d, 1),
            bias_attr=init_params('TEM_e2_b', self.hidden_dim_1d, 1))
        self.te_conv2_act = paddle.nn.Sigmoid()

        #Proposal Evaluation Module
        self.p_conv1 = paddle.nn.Conv1D(
            in_channels=self.hidden_dim_1d,
            out_channels=self.hidden_dim_2d,
            kernel_size=3,
            padding=1,
            groups=1,
            weight_attr=init_params('PEM_1d_w', self.hidden_dim_1d, 3),
            bias_attr=init_params('PEM_1d_b', self.hidden_dim_1d, 3))
        self.p_conv1_act = paddle.nn.ReLU()

        # init to speed up
        sample_mask = get_interp1d_mask(self.tscale, self.dscale,
                                        self.prop_boundary_ratio,
                                        self.num_sample, self.num_sample_perbin)
        self.sample_mask = paddle.to_tensor(sample_mask)
        self.sample_mask.stop_gradient = True

        self.p_conv3d1 = paddle.nn.Conv3D(
            in_channels=128,
            out_channels=self.hidden_dim_3d,
            kernel_size=(self.num_sample, 1, 1),
            stride=(self.num_sample, 1, 1),
            padding=0,
            weight_attr=ParamAttr(name="PEM_3d1_w"),
            bias_attr=ParamAttr(name="PEM_3d1_b"))
        self.p_conv3d1_act = paddle.nn.ReLU()

        self.p_conv2d1 = paddle.nn.Conv2D(
            in_channels=512,
            out_channels=self.hidden_dim_2d,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name="PEM_2d1_w"),
            bias_attr=ParamAttr(name="PEM_2d1_b"))
        self.p_conv2d1_act = paddle.nn.ReLU()

        self.p_conv2d2 = paddle.nn.Conv2D(
            in_channels=128,
            out_channels=self.hidden_dim_2d,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(name="PEM_2d2_w"),
            bias_attr=ParamAttr(name="PEM_2d2_b"))
        self.p_conv2d2_act = paddle.nn.ReLU()

        self.p_conv2d3 = paddle.nn.Conv2D(
            in_channels=128,
            out_channels=self.hidden_dim_2d,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(name="PEM_2d3_w"),
            bias_attr=ParamAttr(name="PEM_2d3_b"))
        self.p_conv2d3_act = paddle.nn.ReLU()

        self.p_conv2d4 = paddle.nn.Conv2D(
            in_channels=128,
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name="PEM_2d4_w"),
            bias_attr=ParamAttr(name="PEM_2d4_b"))
        self.p_conv2d4_act = paddle.nn.Sigmoid()

    def init_weights(self):
        pass

    def forward(self, x):

        alpha = 0.9

        #Base Module
        x = self.b_conv1(x)
        x = self.b_conv1_act(x)
        x = self.b_conv2(x)
        x = self.b_conv2_act(x) # 2 256 300

        m = self.lgte1(x)

        #SE Block
        if self.training:
            x = x
        else:
            x = paddle.fluid.layers.reshape(x, (1, 256, 300))
        b = x.shape[0]
        c = x.shape[1]
        y = self.se_avg_pool(x)
        y = paddle.fluid.layers.reshape(y, (b, c))
        y = self.se_linear_1(y)
        y = self.se_act_1(y)
        y = self.se_linear_2(y)
        y = self.se_act_2(y)
        y = paddle.fluid.layers.reshape(y, (b, c, 1))
        y = paddle.expand_as(y, x)
        z = x * y
        x = x + z
        if self.training:
            x = x
        else:
            x = paddle.fluid.layers.reshape(x, (-1, 256, 300))

        x = alpha * x + (1-alpha) * m

        #TEM
        xs = self.ts_conv1(x)
        xs = self.ts_conv1_act(xs)
        xs = self.ts_conv2(xs)
        xs = self.ts_conv2_act(xs)
        xs = paddle.squeeze(xs, axis=[1])
        xe = self.te_conv1(x)
        xe = self.te_conv1_act(xe)
        xe = self.te_conv2(xe)
        xe = self.te_conv2_act(xe)
        xe = paddle.squeeze(xe, axis=[1])

        #PEM
        xp = self.p_conv1(x)
        xp = self.p_conv1_act(xp)
        #BM layer
        xp = paddle.matmul(xp, self.sample_mask)
        xp = paddle.reshape(xp, shape=[0, 0, -1, self.dscale, self.tscale])

        xp = self.p_conv3d1(xp)
        xp = self.p_conv3d1_act(xp)
        xp = paddle.squeeze(xp, axis=[2])

        xp = self.p_conv2d1(xp)
        xp = self.p_conv2d1_act(xp)

        xp = self.p_conv2d2(xp)
        xp = self.p_conv2d2_act(xp)
        xp = self.p_conv2d3(xp)
        xp = self.p_conv2d3_act(xp)
        xp = self.p_conv2d4(xp)
        xp = self.p_conv2d4_act(xp)
        return xp, xs, xe
