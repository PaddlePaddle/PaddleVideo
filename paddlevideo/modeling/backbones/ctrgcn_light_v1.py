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
from ..registry import BACKBONES
from ..weight_init import weight_init_


def conv_init(conv):
    if conv.weight is not None:
        weight_init_(conv.weight, 'kaiming_normal_', mode='fan_in')
    if conv.bias is not None:
        nn.initializer.Constant(value=0.0)(conv.bias)


def bn_init(bn, scale):
    nn.initializer.Constant(value=float(scale))(bn.weight)
    nn.initializer.Constant(value=0.0)(bn.bias)


def einsum(x1, x3):
    """paddle.einsum only support in dynamic graph mode.
    x1 : n c u v
    x2 : n c t v
    """
    n, c, u, v1 = x1.shape
    n, c, t, v3 = x3.shape
    assert (v1 == v3), "Args of einsum not match!"
    x1 = paddle.transpose(x1, perm=[0, 1, 3, 2])  # n c v u
    y = paddle.matmul(x3, x1)
    # out: n c t u
    return y

class TemporalConv(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0),
                              stride=(stride, 1),
                              dilation=(dilation, 1),
                              bias_attr=False)

        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super(MultiScale_TemporalConv, self).__init__()
        assert out_channels % (
            len(dilations) +
            2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        
        self.conv_pw = nn.Sequential(
                            nn.Conv2D(in_channels,
                                        branch_channels*(self.num_branches-1),
                                        kernel_size=1,
                                        padding=0,
                                        bias_attr=False), 
                            nn.BatchNorm2D(branch_channels*(self.num_branches-1)),
                            nn.ReLU(),
        )
        # Temporal Convolution branches
        self.branches = nn.LayerList([
            TemporalConv(branch_channels,
                            branch_channels,
                            kernel_size=ks,
                            stride=stride,
                            dilation=dilation
            ) for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(
            nn.Sequential(
                nn.MaxPool2D(kernel_size=(3, 1),
                             stride=(stride, 1),
                             padding=(1, 0)), 
                nn.BatchNorm2D(branch_channels)))

        self.branches.append(
            nn.Sequential(
                nn.Conv2D(in_channels,
                          branch_channels,
                          kernel_size=1,
                          padding=0,
                          stride=(stride, 1),
                          bias_attr=False), nn.BatchNorm2D(branch_channels)))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels,
                                         out_channels,
                                         kernel_size=residual_kernel_size,
                                         stride=stride)

    def init_weights(self):
        """Initiate the parameters.
        """
        # initialize
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2D):
                weight_init_(m.weight, 'Normal', std=0.02, mean=1.0)
                nn.initializer.Constant(value=0.0)(m.bias)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        input_x = paddle.chunk(self.conv_pw(x), chunks=self.num_branches-1, axis=1)
        input_x.append(x)
        branch_outs = []
        for tempconv, _ in zip(self.branches, input_x):
            out = tempconv(_)
            branch_outs.append(out)

        out = paddle.concat(branch_outs, axis=1)
        out += res
        return out

class CTRGC(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_subset,
                 rel_reduction=8,
                 mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subset = num_subset
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2D(self.in_channels,
                               self.rel_channels*2*self.num_subset+self.out_channels*self.num_subset,
                               kernel_size=1)
        self.conv2 = nn.LayerList()
        for i in range(self.num_subset):
            self.conv2.append(nn.Conv2D(self.rel_channels,
                               self.out_channels,
                               kernel_size=1))

        self.tanh = nn.Tanh()

        self.out_channels_ns = self.out_channels*self.num_subset

    def init_weights(self):
        """Initiate the parameters.
        """
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2D):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x = self.conv1(x)
        x1, x2 = paddle.chunk(x[:, :-self.out_channels_ns], chunks=2, axis=1)
        x1 = self.tanh(x1.mean(-2).unsqueeze(-1) - x2.mean(-2).unsqueeze(-2))
        x1 = paddle.chunk(x1, chunks=self.num_subset, axis=1)
        x2 = paddle.chunk(x[:, -self.out_channels_ns:], chunks=self.num_subset, axis=1)
        y = None
        for i in range(self.num_subset):
            z =  self.conv2[i](x1[i])* alpha + (A[i].unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
            # We only support 'paddle.einsum()' in dynamic graph mode, if use in infer model please implement self.
            # x1 = paddle.einsum('ncuv,nctv->nctu', x1, x3)
            z = einsum(z, x2[i])
            y = z + y if y is not None else z
        return y

class unit_gcn(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 coff_embedding=4,
                 adaptive=True,
                 residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.LayerList()

        self.convs = CTRGC(in_channels, out_channels, num_subset=self.num_subset)

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
                    nn.BatchNorm2D(out_channels))
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            pa_param = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Assign(A.astype(np.float32)))
            self.PA = paddle.create_parameter(shape=A.shape,
                                              dtype='float32',
                                              attr=pa_param)
        else:
            A_tensor = paddle.to_tensor(A, dtype="float32")
            self.A = paddle.create_parameter(
                shape=A_tensor.shape,
                dtype='float32',
                default_initializer=paddle.nn.initializer.Assign(A_tensor))
            self.A.stop_gradient = True
        alpha_tensor = paddle.to_tensor(np.zeros(1), dtype="float32")
        self.alpha = paddle.create_parameter(
            shape=alpha_tensor.shape,
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(alpha_tensor))
        self.bn = nn.BatchNorm2D(out_channels)
        #self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

    def init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2D):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        y = self.convs(x, A, self.alpha)
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y


class TCN_GCN_unit(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 adaptive=True,
                 kernel_size=5,
                 dilations=[1, 2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = TemporalConv(in_channels,
                                     out_channels,
                                     kernel_size=1,
                                     stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

class TCN_unit(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 residual=True,
                 kernel_size=5,
                 dilations=[1, 2]):
        super(TCN_unit, self).__init__()
        self.tcn1 = MultiScale_TemporalConv(in_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            dilations=dilations,
                                            residual=residual)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.tcn1(x))
        return y

class NTUDGraph:

    def __init__(self, labeling_mode='spatial'):
        num_node = 25
        self_link = [(i, i) for i in range(num_node)]
        inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                            (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
                            (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                            (17, 1), (18, 17), (19, 18), (20, 19), (22, 23),
                            (23, 8), (24, 25), (25, 12)]
        inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward

        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def edge2mat(self, link, num_node):
        A = np.zeros((num_node, num_node))
        for i, j in link:
            A[j, i] = 1
        return A

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        h, w = A.shape
        Dn = np.zeros((w, w))
        for i in range(w):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD

    def get_spatial_graph(self, num_node, self_link, inward, outward):
        I = self.edge2mat(self_link, num_node)
        In = self.normalize_digraph(self.edge2mat(inward, num_node))
        Out = self.normalize_digraph(self.edge2mat(outward, num_node))
        A = np.stack((I, In, Out))
        return A

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = self.get_spatial_graph(self.num_node, self.self_link,
                                       self.inward, self.outward)
        else:
            raise ValueError()
        return A


@BACKBONES.register()
class CTRGCN_lightV1(nn.Layer):
    """
    CTR-GCN model from:
    `"Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition" <https://arxiv.org/abs/2107.12213>`_
    Args:
        num_point: int, numbers of sketeton point.
        num_person: int, numbers of person.
        base_channel: int, model's hidden dim.
        graph: str, sketeton adjacency matrix name.
        graph_args: dict, sketeton adjacency graph class args.
        in_channels: int, channels of vertex coordinate. 2 for (x,y), 3 for (x,y,z). Default 3.
        adaptive: bool, if adjacency matrix can adaptive.
    """

    def __init__(self,
                 num_point=25,
                 num_person=2,
                 base_channel=64,
                 graph='ntu_rgb_d',
                 graph_args=dict(),
                 in_channels=3,
                 adaptive=True):
        super(CTRGCN_lightV1, self).__init__()

        if graph == 'ntu_rgb_d':
            self.graph = NTUDGraph(**graph_args)
        else:
            raise ValueError()

        A = self.graph.A  # 3,25,25

        self.num_point = num_point
        self.data_bn = nn.BatchNorm1D(num_person * in_channels * num_point)
        self.base_channel = base_channel

        self.l1 = TCN_GCN_unit(in_channels,
                               self.base_channel,
                               A,
                               residual=False,
                               adaptive=adaptive)
        self.l2 = TCN_unit(self.base_channel,
                               self.base_channel)
        self.l3 = TCN_unit(self.base_channel,
                               self.base_channel)
        self.l4 = TCN_GCN_unit(self.base_channel,
                               self.base_channel,
                               A,
                               adaptive=adaptive)
        self.l5 = TCN_unit(self.base_channel,
                               self.base_channel * 2,
                               stride=2)
        self.l6 = TCN_GCN_unit(self.base_channel * 2,
                               self.base_channel * 2,
                               A,
                               adaptive=adaptive)
        self.l7 = TCN_GCN_unit(self.base_channel * 2,
                               self.base_channel * 2,
                               A,
                               adaptive=adaptive)
        self.l8 = TCN_unit(self.base_channel * 2,
                               self.base_channel * 4,
                               stride=2)
        self.l9 = TCN_GCN_unit(self.base_channel * 4,
                               self.base_channel * 4,
                               A,
                               adaptive=adaptive)
        self.l10 = TCN_GCN_unit(self.base_channel * 4,
                                self.base_channel * 4,
                                A,
                                adaptive=adaptive)

    def init_weights(self):
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.shape
        x = paddle.transpose(x, perm=[0, 4, 3, 1, 2])
        x = paddle.reshape(x, (N, M * V * C, T))

        x = self.data_bn(x)

        x = paddle.reshape(x, (N, M, V, C, T))
        x = paddle.transpose(x, perm=(0, 1, 3, 4, 2))

        x = paddle.reshape(x, (N * M, C, T, V))

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        return x, N, M
