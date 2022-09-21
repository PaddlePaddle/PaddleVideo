# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

import copy as cp

import paddle
import paddle.nn as nn
import numpy as np

from .stgcn import Graph as GraphBase, normalize_digraph
from ..weight_init import kaiming_normal_, constant_
from ..registry import BACKBONES

EPS = 1e-4


class Graph(GraphBase):

    def edge2mat(self, link, num_node):
        A = np.zeros((num_node, num_node))
        for i, j in link:
            A[j, i] = 1
        return A

    def get_edge(self, layout):
        # edge is a list of [child, parent] paris

        if layout == 'coco_keypoint':
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5),
                              (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0),
                              (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
            neighbor_link = [(i, j) for (i, j) in neighbor_1base]
            self.outward = [(j, i) for (i, j) in neighbor_1base]
            self.self_link = self_link
            self.neighbor_1base = neighbor_1base
            self.edge = self_link + neighbor_link
            self.center = 0
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1

        if strategy == 'spatial':
            Iden = self.edge2mat(self.self_link, self.num_node)
            In = normalize_digraph(
                self.edge2mat(self.neighbor_1base, self.num_node))
            Out = normalize_digraph(self.edge2mat(self.outward, self.num_node))
            A = np.stack((Iden, In, Out))
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


class mstcn(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(
                    nn.Conv2D(in_channels,
                              branch_c,
                              kernel_size=1,
                              stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2D(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2D(branch_c), self.act,
                        nn.MaxPool2D(kernel_size=(cfg[1], 1),
                                     stride=(stride, 1),
                                     padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2D(in_channels, branch_c, kernel_size=1),
                nn.BatchNorm2D(branch_c), self.act,
                unit_tcn(branch_c,
                         branch_c,
                         kernel_size=cfg[0],
                         stride=stride,
                         dilation=cfg[1],
                         norm=None))
            branches.append(branch)

        self.branches = nn.LayerList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2D(tin_channels), self.act,
            nn.Conv2D(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2D(out_channels)
        self.drop = nn.Dropout(dropout)

    def inner_forward(self, x):
        N, C, T, V = x.shape

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        feat = paddle.concat(branch_outs, axis=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)


class unit_tcn(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=9,
                 stride=1,
                 dilation=1,
                 norm='BN',
                 dropout=0):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0),
                              stride=(stride, 1),
                              dilation=(dilation, 1))
        self.bn = nn.BatchNorm2D(
            out_channels) if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))

    def init_weights(self):
        kaiming_normal_(self.conv.weight, mode='fan_out')
        nn.initializer.Constant(0)(self.conv.bias)
        nn.initializer.Constant(1)(self.bn.weight)
        nn.initializer.Constant(0)(self.bn.bias)


class unit_gcn(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 adaptive='importance',
                 conv_pos='pre',
                 with_res=False,
                 norm='BN',
                 act='ReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.shape[0]

        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive
        assert conv_pos in ['pre', 'post']
        self.conv_pos = conv_pos
        self.with_res = with_res

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.bn = nn.BatchNorm2D(out_channels)
        self.act = nn.ReLU()

        if self.adaptive == 'init':
            self.A = paddle.to_tensor(A.clone())
        else:
            self.register_buffer('A', A)

        if self.adaptive in ['offset', 'importance']:
            self.PA = paddle.to_tensor(A.clone())
            if self.adaptive == 'offset':
                nn.initializer.Uniform(-1e-6, 1e-6)(self.PA)
            elif self.adaptive == 'importance':
                nn.initializer.Constant(1)(self.PA)

        if self.conv_pos == 'pre':
            self.conv = nn.Conv2D(in_channels, out_channels * A.shape[0], 1)
        elif self.conv_pos == 'post':
            self.conv = nn.Conv2D(A.shape[0] * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2D(in_channels, out_channels, 1),
                    nn.BatchNorm2D(out_channels))
            else:
                self.down = lambda x: x

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0

        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({
                'offset': self.A + self.PA,
                'importance': self.A * self.PA
            })
        A = A_switch[self.adaptive]

        if self.conv_pos == 'pre':
            x = self.conv(x)
            x = x.reshape(
                [n, self.num_subsets, x.shape[1] // self.num_subsets, t, v])
            x = paddle.einsum('nkctv,kvw->nctw', x, A)
        elif self.conv_pos == 'post':
            x = paddle.einsum('nctv,kvw->nkctw', x, A)
            x = x.reshape([n, -1, t, v])
            x = self.conv(x)

        return self.act(self.bn(x) + res)


class STGCNBlock(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 **kwargs):
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {
            k: v
            for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']
        }
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)

        self.tcn = mstcn(out_channels,
                         out_channels,
                         stride=stride,
                         **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels,
                                     out_channels,
                                     kernel_size=1,
                                     stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


@BACKBONES.register()
class STGCNPlusPlus(nn.Layer):

    def __init__(
            self,
            in_channels=3,
            base_channels=64,
            data_bn_type='VC',
            ch_ratio=2,
            num_person=2,  # * Only used when data_bn_type == 'MVC'
            num_stages=10,
            inflate_stages=[5, 8],
            down_stages=[5, 8],
            pretrained=None,
            layout='coco',
            strategy='spatial',
            **kwargs):
        super().__init__()

        self.graph = Graph(layout=layout, strategy=strategy)

        A = paddle.to_tensor(self.graph.A, dtype='float32')

        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1D(num_person * in_channels * A.shape[1])
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1D(in_channels * A.shape[1])
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [
                STGCNBlock(in_channels,
                           base_channels,
                           A.clone(),
                           1,
                           residual=False,
                           **lw_kwargs[0])
            ]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels *
                               self.ch_ratio**inflate_times + EPS)
            base_channels = out_channels
            modules.append(
                STGCNBlock(in_channels, out_channels, A.clone(), stride,
                           **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.LayerList(modules)
        self.pretrained = pretrained
        self.init_weights()

    def init_weights(self):
        for i, m in enumerate(self.sublayers()):
            if isinstance(m, paddle.nn.Linear):
                kaiming_normal_(m.weight, reverse=True)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, paddle.nn.Conv2D):
                kaiming_normal_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, paddle.nn.Conv1D):
                kaiming_normal_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, paddle.nn.BatchNorm2D):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
            elif isinstance(m, paddle.nn.BatchNorm1D):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        bs, nc = x.shape[:2]
        x = x.reshape([bs * nc] + list(x.shape[2:]))
        N, M, T, V, C = x.shape
        x = paddle.transpose(x, [0, 1, 3, 4, 2])
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.reshape([N, M * V * C, T]))
        else:
            x = self.data_bn(x.reshape([-1, V * C, T]))
        x = x.reshape([N, M, V, C, T]).transpose([0, 1, 3, 4,
                                                  2]).reshape([-1, C, T, V])

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape([N, M] + list(x.shape[1:]))
        return x
