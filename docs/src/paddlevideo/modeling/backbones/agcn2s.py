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
import numpy as np
from ..registry import BACKBONES


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class UnitTCN(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(UnitTCN, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        " input size : (N*M, C, T, V)"
        x = self.bn(self.conv(x))
        return x


class UnitGCN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 coff_embedding=4,
                 num_subset=3):
        super(UnitGCN, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        PA = self.create_parameter(shape=A.shape, dtype='float32')
        self.PA = PA
        self.A = paddle.to_tensor(A.astype(np.float32))
        self.num_subset = num_subset

        self.conv_a = nn.LayerList()
        self.conv_b = nn.LayerList()
        self.conv_d = nn.LayerList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2D(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2D(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2D(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2D(in_channels, out_channels, 1),
                                      nn.BatchNorm2D(out_channels))
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2D(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

    def forward(self, x):
        N, C, T, V = x.shape
        A = self.A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = paddle.transpose(self.conv_a[i](x),
                                  perm=[0, 3, 1,
                                        2]).reshape([N, V, self.inter_c * T])
            A2 = self.conv_b[i](x).reshape([N, self.inter_c * T, V])
            A1 = self.soft(paddle.matmul(A1, A2) / A1.shape[-1])
            A1 = A1 + A[i]
            A2 = x.reshape([N, C * T, V])
            z = self.conv_d[i](paddle.matmul(A2, A1).reshape([N, C, T, V]))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class Block(nn.Layer):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(Block, self).__init__()
        self.gcn1 = UnitGCN(in_channels, out_channels, A)
        self.tcn1 = UnitTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = UnitTCN(in_channels,
                                    out_channels,
                                    kernel_size=1,
                                    stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


# This Graph structure is for the NTURGB+D dataset. If you use a custom dataset, modify num_node and the corresponding graph adjacency structure.
class Graph:
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
class AGCN2s(nn.Layer):
    def __init__(self,
                 num_point=25,
                 num_person=2,
                 graph='ntu_rgb_d',
                 graph_args=dict(),
                 in_channels=3):
        super(AGCN2s, self).__init__()

        if graph == 'ntu_rgb_d':
            self.graph = Graph(**graph_args)
        else:
            raise ValueError()

        A = self.graph.A
        self.data_bn = nn.BatchNorm1D(num_person * in_channels * num_point)

        self.l1 = Block(in_channels, 64, A, residual=False)
        self.l2 = Block(64, 64, A)
        self.l3 = Block(64, 64, A)
        self.l4 = Block(64, 64, A)
        self.l5 = Block(64, 128, A, stride=2)
        self.l6 = Block(128, 128, A)
        self.l7 = Block(128, 128, A)
        self.l8 = Block(128, 256, A, stride=2)
        self.l9 = Block(256, 256, A)
        self.l10 = Block(256, 256, A)

    def forward(self, x):
        N, C, T, V, M = x.shape

        x = x.transpose([0, 4, 3, 1, 2]).reshape_([N, M * V * C, T])
        x = self.data_bn(x)
        x = x.reshape_([N, M, V, C,
                        T]).transpose([0, 1, 3, 4,
                                       2]).reshape_([N * M, C, T, V])

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

        return x
