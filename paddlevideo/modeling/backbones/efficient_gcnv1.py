import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..registry import BACKBONES
import math
import numpy as np
import paddle.nn.initializer as init
from ..weight_init import weight_init_


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Graph():
    def __init__(self, dataset, max_hop=10, dilation=1):
        self.dataset = dataset.split('-')[0]
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge(
        )

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.dataset == 'kinetics':
            num_node = 18
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14),
                             (8, 11)]
            connect_joint = np.array(
                [1, 1, 1, 2, 3, 1, 5, 6, 2, 8, 9, 5, 11, 12, 0, 0, 14, 15])
            parts = [
                np.array([5, 6, 7]),  # left_arm
                np.array([2, 3, 4]),  # right_arm
                np.array([11, 12, 13]),  # left_leg
                np.array([8, 9, 10]),  # right_leg
                np.array([0, 1, 14, 15, 16, 17])  # torso
            ]
        elif self.dataset == 'ntu':
            num_node = 25
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                              (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
                              (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                              (17, 1), (18, 17), (19, 18), (20, 19), (22, 23),
                              (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([
                2, 2, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17,
                18, 19, 2, 23, 8, 25, 12
            ]) - 1
            parts = [
                np.array([5, 6, 7, 8, 22, 23]) - 1,  # left_arm
                np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
                np.array([13, 14, 15, 16]) - 1,  # left_leg
                np.array([17, 18, 19, 20]) - 1,  # right_leg
                np.array([1, 2, 3, 4, 21]) - 1  # torso
            ]
        elif self.dataset == 'sysu':
            num_node = 20
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (6, 7),
                              (7, 8), (3, 9), (9, 10), (10, 11), (11, 12),
                              (1, 13), (13, 14), (14, 15), (15, 16), (1, 17),
                              (17, 18), (18, 19), (19, 20)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([
                2, 2, 2, 3, 3, 5, 6, 7, 3, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18,
                19
            ]) - 1
            parts = [
                np.array([5, 6, 7, 8]) - 1,  # left_arm
                np.array([9, 10, 11, 12]) - 1,  # right_arm
                np.array([13, 14, 15, 16]) - 1,  # left_leg
                np.array([17, 18, 19, 20]) - 1,  # right_leg
                np.array([1, 2, 3, 4]) - 1  # torso
            ]
        elif self.dataset == 'ucla':
            num_node = 20
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (6, 7),
                              (7, 8), (3, 9), (9, 10), (10, 11), (11, 12),
                              (1, 13), (13, 14), (14, 15), (15, 16), (1, 17),
                              (17, 18), (18, 19), (19, 20)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([
                2, 2, 2, 3, 3, 5, 6, 7, 3, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18,
                19
            ]) - 1
            parts = [
                np.array([5, 6, 7, 8]) - 1,  # left_arm
                np.array([9, 10, 11, 12]) - 1,  # right_arm
                np.array([13, 14, 15, 16]) - 1,  # left_leg
                np.array([17, 18, 19, 20]) - 1,  # right_leg
                np.array([1, 2, 3, 4]) - 1  # torso
            ]
        elif self.dataset == 'cmu':
            num_node = 26
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
                              (1, 9), (5, 9), (9, 10), (10, 11), (11, 12),
                              (12, 13), (13, 14), (12, 15), (15, 16), (16, 17),
                              (17, 18), (18, 19), (17, 20), (12, 21), (21, 22),
                              (22, 23), (23, 24), (24, 25), (23, 26)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([
                9, 1, 2, 3, 9, 5, 6, 7, 10, 10, 10, 11, 12, 13, 12, 15, 16, 17,
                18, 17, 12, 21, 22, 23, 24, 23
            ]) - 1
            parts = [
                np.array([15, 16, 17, 18, 19, 20]) - 1,  # left_arm
                np.array([21, 22, 23, 24, 25, 26]) - 1,  # right_arm
                np.array([1, 2, 3, 4]) - 1,  # left_leg
                np.array([5, 6, 7, 8]) - 1,  # right_leg
                np.array([9, 10, 11, 12, 13, 14]) - 1  # torso
            ]
        elif self.dataset == 'h36m':
            num_node = 20
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
                              (1, 9), (5, 9), (9, 10), (10, 11), (11, 12),
                              (10, 13), (13, 14), (14, 15), (15, 16), (10, 17),
                              (17, 18), (18, 19), (19, 20)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([
                9, 1, 2, 3, 9, 5, 6, 7, 9, 9, 10, 11, 10, 13, 14, 15, 10, 17,
                18, 19
            ]) - 1
            parts = [
                np.array([13, 14, 15, 16]) - 1,  # left_arm
                np.array([17, 18, 19, 20]) - 1,  # right_arm
                np.array([1, 2, 3, 4]) - 1,  # left_leg
                np.array([5, 6, 7, 8]) - 1,  # right_leg
                np.array([9, 10, 11, 12]) - 1  # torso
            ]
        else:
            raise ValueError()
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [
            np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)
        ]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD


## activation
class Swish(nn.Layer):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x = x.multiply(F.sigmoid(x))
            return x
        else:
            return x.multiply(F.sigmoid(x))


class HardSwish(nn.Layer):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        inner = F.relu6(x + 3.).divide(6.)
        if self.inplace:
            x = x.multiply(inner)
            return x
        else:
            return x.multiply(inner)


class AconC(nn.Layer):
    def __init__(self, channel):
        super(AconC, self).__init__()
        p1 = paddle.randn([1, channel, 1, 1], dtype="float32")

        p1 = self.create_parameter(
            shape=p1.shape,
            dtype=str(p1.numpy().dtype),
            attr=paddle.ParamAttr(
                initializer=nn.initializer.Assign(p1), trainable=True))
        self.add_parameter('p1', p1)
        p2 = paddle.randn([1, channel, 1, 1], dtype="float32")
        p2 = self.create_parameter(
            shape=p2.shape,
            dtype=str(p2.numpy().dtype),
            attr=paddle.ParamAttr(
                initializer=nn.initializer.Assign(p2), trainable=True))
        self.add_parameter('p2', p1)
        beta = paddle.ones([1, channel, 1, 1], dtype="float32")
        beta = self.create_parameter(
            shape=beta.shape,
            dtype=str(beta.numpy().dtype),
            attr=paddle.ParamAttr(
                initializer=nn.initializer.Assign(beta), trainable=True))
        self.add_parameter('beta', beta)

    def forward(self, x):
        # return (self.params[0] * x - self.params[1] * x) * F.simgoid(self.params[2] * (self.params[0] * x - self.params[1] * x)) + self.params[1] * x
        return (self.p1 * x - self.p2 * x) * F.sigmoid(
            self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x


class MetaAconC(nn.Layer):
    def __init__(self, channel, r=4):
        super(MetaAconC, self).__init__()
        inner_channel = max(r, channel // r)

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channel, inner_channel, 1, stride=(1, 1)),
            nn.BatchNorm2D(inner_channel),
            nn.Conv2D(inner_channel, channel, 1, stride=(1, 1)),
            nn.BatchNorm2D(channel),
            nn.Sigmoid(),
        )
        p1 = paddle.randn([1, channel, 1, 1], dtype='float32')
        p2 = paddle.randn([1, channel, 1, 1], dtype='float32')
        p1 = self.create_parameter(
            shape=p1.shape,
            dtype=str(p1.numpy().dtype),
            attr=paddle.ParamAttr(
                initializer=nn.initializer.Assign(p1), trainable=True))
        self.add_parameter('p1', p1)
        p2 = self.create_parameter(
            shape=p2.shape,
            dtype=str(p2.numpy().dtype),
            attr=paddle.ParamAttr(
                initializer=nn.initializer.Assign(p2), trainable=True))
        self.add_parameter('p2', p2)

    def forward(self, x, **kwargs):
        return (self.p1 * x - self.p2 * x) * F.sigmoid(
            self.fcn(x) * (self.p1 * x - self.p2 * x)) + self.p2 * x


activation = {
    'swish': Swish,
    'hswish': HardSwish,
    'aconc': AconC,
    'metaaconc': MetaAconC
}


## attentions
class Attention_Layer(nn.Layer):
    def __init__(self, out_channel, att_type, act, **kwargs):
        super(Attention_Layer, self).__init__()

        __attetion = {
            'stja': ST_Joint_Att,
            'pa': Part_Att,
            'ca': Channel_Att,
            'fa': Frame_Att,
            'ja': Joint_Att,
        }

        self.att = __attetion[att_type](channel=out_channel, **kwargs)
        self.bn = nn.BatchNorm2D(out_channel)
        self.act = activation[act]()

    def forward(self, x):
        res = x
        x = x * self.att(x)
        return self.act(self.bn(x) + res)


class ST_Joint_Att(nn.Layer):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(ST_Joint_Att, self).__init__()
        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2D(
                channel,
                inner_channel,
                kernel_size=1,
                stride=(1, 1),
                bias_attr=bias),
            nn.BatchNorm2D(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2D(
            inner_channel, channel, kernel_size=1, stride=(1, 1))
        self.conv_v = nn.Conv2D(
            inner_channel, channel, kernel_size=1, stride=(1, 1))

    def forward(self, x):
        N, C, T, V = x.shape
        x_t = paddle.mean(x, axis=3, keepdim=True)
        x_v = paddle.mean(x, axis=2, keepdim=True)
        x_v = x_v.transpose((0, 1, 3, 2))
        x_att = self.fcn(paddle.concat(x=[x_t, x_v], axis=2))
        x_t, x_v = paddle.split(x_att, [T, V], axis=2)

        x_t_att = F.sigmoid(self.conv_t(x_t))

        x_v_att = F.sigmoid(self.conv_v(x_v.transpose((0, 1, 3, 2))))
        x_att = x_t_att * x_v_att
        return x_att


class Part_Att(nn.Layer):
    def __init__(self, channel, parts, reduct_ratio, bias, A, **kwargs):
        super(Part_Att, self).__init__()

        self.parts = parts
        joints = self.get_corr_joints()
        joints = self.create_parameter(
            shape=A.shape,
            dtype=str(joints.numpy().dtype),
            attr=paddle.ParamAttr(
                initializer=nn.initializer.Assign(joints), trainable=False))
        self.add_parameter('joints', joints)

        inner_channel = channel // reduct_ratio

        self.softmax = nn.Softmax(axis=3)
        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(
                channel,
                inner_channel,
                kernel_size=1,
                stride=(1, 1),
                bias_attr=bias), nn.BatchNorm2D(inner_channel), nn.ReLU(),
            nn.Conv2D(
                inner_channel,
                channel * len(self.parts),
                kernel_size=1,
                stride=(1, 1),
                bias_attr=bias))

    def forward(self, x):
        N, C, T, V = x.shape
        x_att = self.softmax(self.fcn(x).view(N, C, 1, len(self.parts)))
        x_att = x_att.index_select(3, self.joints).expand_as(x)
        return x_att

    def get_corr_joints(self):
        num_joints = sum([len(part) for part in self.parts])
        joints = [
            j for i in range(num_joints) for j in range(len(self.parts))
            if i in self.parts[j]
        ]
        return paddle.to_tensor(joints, dtype="float64")


class Channel_Att(nn.Layer):
    def __init__(self, channel, **kwargs):
        super(Channel_Att, self).__init__()

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channel, channel // 4, kernel_size=1, stride=(1, 1)),
            nn.BatchNorm2D(channel // 4), nn.ReLU(),
            nn.Conv2D(channel // 4, channel, kernel_size=1, stride=(1, 1)),
            nn.Sigmoid())

    def forward(self, x):
        return self.fcn(x)


class Frame_Att(nn.Layer):
    def __init__(self, **kwargs):
        super(Frame_Att, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)

        self.conv = nn.Conv2D(
            2, 1, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))

    def forward(self, x):
        x = x.transpose((0, 2, 1, 3))
        x = paddle.concat([self.avg_pool(x), self.max_pool(x)],
                          axis=2).transpose([0, 2, 1, 3])
        return self.conv(x)


class Joint_Att(nn.Layer):
    def __init__(self, parts, **kwargs):
        super(Joint_Att, self).__init__()

        num_joints = sum([len(part) for part in parts])
        self.fcn = nn.Sequential(
            nn.AdaptiveMaxPool2D(1),
            nn.Conv2D(
                num_joints, num_joints // 2, kernel_size=1, stride=(1, 1)),
            nn.BatchNorm2D(num_joints // 2), nn.ReLU(),
            nn.Conv2D(
                num_joints // 2, num_joints, kernel_size=1, stride=(1, 1)),
            nn.Softmax(axis=1))

    def forward(self, x):
        return self.fcn(x.transpose((0, 3, 2, 1))).transpose((0, 3, 2, 1))


class Basic_Layer(nn.Layer):
    def __init__(self, in_channel, out_channel, residual, bias, act, **kwargs):
        super(Basic_Layer, self).__init__()

        self.conv = nn.Conv2D(
            in_channel,
            out_channel,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias_attr=bias)
        self.bn = nn.BatchNorm2D(out_channel)

        self.residual = nn.Identity() if residual else Zero_Layer()
        self.act = activation[act]()

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.bn(self.conv(x)) + res)
        return x


class Spatial_Graph_Layer(Basic_Layer):
    def __init__(self,
                 in_channel,
                 out_channel,
                 max_graph_distance,
                 bias,
                 residual=True,
                 **kwargs):
        super(Spatial_Graph_Layer, self).__init__(in_channel, out_channel,
                                                  residual, bias, **kwargs)
        self.conv = SpatialGraphConv(in_channel, out_channel,
                                     max_graph_distance, bias, **kwargs)
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2D(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=(1, 1),
                    stride=[1, 1],
                    bias_attr=bias),
                nn.BatchNorm2D(out_channel),
            )


class Temporal_Basic_Layer(Basic_Layer):
    def __init__(self,
                 channel,
                 temporal_window_size,
                 bias,
                 stride=1,
                 residual=True,
                 **kwargs):
        super(Temporal_Basic_Layer, self).__init__(channel, channel, residual,
                                                   bias, **kwargs)

        padding = (temporal_window_size - 1) // 2
        self.conv = nn.Conv2D(
            in_channels=channel,
            out_channels=channel,
            kernel_size=(temporal_window_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            bias_attr=bias)
        if residual and stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2D(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias_attr=bias), nn.BatchNorm2D(channel))


class Temporal_Bottleneck_Layer(nn.Layer):
    def __init__(self,
                 channel,
                 temporal_window_size,
                 bias,
                 act,
                 reduct_ratio,
                 stride=1,
                 residual=True,
                 **kwargs):
        super(Temporal_Bottleneck_Layer, self).__init__()
        inner_channel = channel // reduct_ratio
        padding = (temporal_window_size - 1) // 2
        self.act = activation[act]()
        self.reduct_conv = nn.Sequential(
            nn.Conv2D(
                in_channels=channel,
                out_channels=inner_channel,
                kernel_size=1,
                stride=(1, 1),
                bias_attr=bias),
            nn.BatchNorm2D(inner_channel),
        )

        self.conv = nn.Sequential(
            nn.Conv2D(
                in_channels=inner_channel,
                out_channels=inner_channel,
                kernel_size=(temporal_window_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
                bias_attr=bias),
            nn.BatchNorm2D(inner_channel),
        )
        self.expand_conv = nn.Sequential(
            nn.Conv2D(
                in_channels=inner_channel,
                out_channels=channel,
                kernel_size=1,
                stride=(1, 1),
                bias_attr=bias),
            nn.BatchNorm2D(channel),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2D(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias_attr=bias), nn.BatchNorm2D(channel))

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.reduct_conv(x))
        x = self.act(self.conv(x))
        x = self.act(self.expand_conv(x) + res)
        return x


class Temporal_Sep_Layer(nn.Layer):
    def __init__(self,
                 channel,
                 temporal_window_size,
                 bias,
                 act,
                 expand_ratio,
                 stride=1,
                 residual=True,
                 **kwargs):
        super(Temporal_Sep_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        self.act = activation[act]()

        if expand_ratio > 0:
            inner_channel = channel * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2D(
                    in_channels=channel,
                    out_channels=inner_channel,
                    kernel_size=1,
                    stride=(1, 1),
                    bias_attr=bias),
                nn.BatchNorm2D(inner_channel),
            )
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2D(
                in_channels=inner_channel,
                out_channels=inner_channel,
                kernel_size=(temporal_window_size, 1),
                stride=(stride, 1),
                padding=(padding, 1),
                groups=inner_channel,
                bias_attr=bias,
                weight_attr=nn.initializer.KaimingNormal()),
            nn.BatchNorm2D(inner_channel),
        )

        self.point_conv = nn.Sequential(
            nn.Conv2D(
                in_channels=inner_channel,
                out_channels=channel,
                kernel_size=1,
                stride=(1, 1),
                bias_attr=bias),
            nn.BatchNorm2D(channel),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2D(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias_attr=bias),
                nn.BatchNorm2D(channel),
            )

    def forward(self, x):
        res = self.residual(x)

        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.act(self.depth_conv(x))
        x = self.point_conv(x)
        return x + res


class Temporal_SG_Layer(nn.Layer):
    def __init__(self,
                 channel,
                 temporal_window_size,
                 bias,
                 act,
                 reduct_ratio,
                 stride=1,
                 residual=True,
                 **kwargs):
        super(Temporal_SG_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        inner_channel = channel // reduct_ratio
        self.act = activation[act]()

        self.depth_conv1 = nn.Sequential(
            nn.Conv2D(
                in_channels=channel,
                out_channels=channel,
                kernel_size=(temporal_window_size, 1),
                stride=(1, 1),
                padding=(padding, 0),
                groups=channel,
                bias_attr=bias),
            nn.BatchNorm2D(channel),
        )

        self.point_conv1 = nn.Sequential(
            nn.Conv2D(
                in_channels=channel,
                out_channels=inner_channel,
                kernel_size=1,
                stride=(1, 1),
                bias_attr=bias),
            nn.BatchNorm2D(inner_channel),
        )

        self.point_conv2 = nn.Sequential(
            nn.Conv2D(
                in_channels=inner_channel,
                out_channels=channel,
                kernel_size=1,
                stride=(1, 1),
                bias_attr=bias),
            nn.BatchNorm2D(channel),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2D(
                in_channels=channel,
                out_channels=channel,
                kernel_size=(temporal_window_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
                groups=channel,
                bias_attr=bias),
            nn.BatchNorm2D(channel),
        )
        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2D(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias_attr=bias),
                nn.BatchNorm2D(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.depth_conv1(x))
        x = self.point_conv1(x)
        x = self.act(self.point_conv2(x))
        x = self.depth_conv2(x)
        return x + res


class Zero_Layer(nn.Layer):
    def __init__(self):
        super(Zero_Layer, self).__init__()

    def forward(self, x):
        return 0


class SpatialGraphConv(nn.Layer):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, edge,
                 A, **kwargs):
        super(SpatialGraphConv, self).__init__()

        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2D(
            in_channel,
            out_channel * self.s_kernel_size,
            1,
            stride=(1, 1),
            bias_attr=bias)
        _A = self.create_parameter(
            shape=A.shape,
            dtype='float32',
            attr=paddle.ParamAttr(
                initializer=nn.initializer.Assign(A), trainable=False))
        self.add_parameter('A', _A)

        if edge:
            ones = paddle.ones_like(self.A)
            _edge = self.create_parameter(
                shape=ones.shape,
                dtype='float32',
                attr=paddle.ParamAttr(
                    initializer=nn.initializer.Assign(ones), trainable=True))
            self.add_parameter('edge', _edge)
        else:
            _edge = self.create_parameter(
                shape=[1],
                dtype='float32',
                attr=paddle.ParamAttr(
                    initializer=nn.initializer.Assign(paddle.to_tensor(1)),
                    trainable=True))
            self.add_parameter('edge', _edge)

    def forward(self, x):
        x = self.gcn(x)
        n, kc, t, v = x.shape
        x = x.reshape([n, self.s_kernel_size, kc // self.s_kernel_size, t, v])
        map = self.A * self.edge
        for i in range(x.shape[1]):
            x[:, i, :, :, :] = paddle.matmul(x[:, i, :, :, :], map[i, :, :])
        x = x.sum(1).squeeze(1)
        return x


@BACKBONES.register()
class EfficientGCN(nn.Layer):
    def __init__(self, data_shape, block_args, fusion_stage, stem_channel,
                 graph, **kwargs):
        super(EfficientGCN, self).__init__()

        num_input, num_channel, _, _, _ = data_shape
        g = Graph(**graph)
        A = g.A[:self.s_kernel_size]
        kwargs['A'] = A
        # input branches
        self.input_branches = nn.LayerList([
            EfficientGCN_Blocks(
                init_channel=stem_channel,
                block_args=block_args[:fusion_stage],
                input_channel=num_channel,
                **kwargs) for _ in range(num_input)
        ])

        # main stream
        last_channel = stem_channel if fusion_stage == 0 else block_args[
            fusion_stage - 1][0]
        self.main_stream = EfficientGCN_Blocks(
            init_channel=num_input * last_channel,
            block_args=block_args[fusion_stage:],
            **kwargs)

        # output
        last_channel = num_input * block_args[-1][0] if fusion_stage == len(
            block_args) else block_args[-1][0]
        self.classifier = EfficientGCN_Classifier(last_channel, **kwargs)

        # init parameters
        init_param(self.sublayers())

    def forward(self, x):

        N, I, C, T, V, M = x.shape
        x = x.transpose((1, 0, 5, 2, 3, 4))
        x = x.reshape([I, N * M, C, T, V])
        # input branches
        x = paddle.concat(
            [branch(x[i]) for i, branch in enumerate(self.input_branches)],
            axis=1)
        # main stream
        x = self.main_stream(x)
        # output
        _, C, T, V = x.shape
        feature = x.reshape([N, M, C, T, V]).transpose((0, 2, 3, 4, 1))
        out = self.classifier(feature).reshape([N, -1])

        return out, feature


class EfficientGCN_Blocks(nn.Sequential):
    def __init__(self,
                 init_channel,
                 block_args,
                 layer_type,
                 kernel_size,
                 input_channel=0,
                 **kwargs):
        super(EfficientGCN_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size

        if input_channel > 0:  # if the blocks in the input branches
            self.add_sublayer('init_bn', nn.BatchNorm2D(input_channel))
            self.add_sublayer(
                'stem_scn',
                Spatial_Graph_Layer(input_channel, init_channel,
                                    max_graph_distance, **kwargs))
            self.add_sublayer(
                'stem_tcn',
                Temporal_Basic_Layer(init_channel, temporal_window_size,
                                     **kwargs))

        last_channel = init_channel
        temporal_layer = None
        if layer_type == 'SG':
            temporal_layer = Temporal_SG_Layer
        elif layer_type == 'Sep':
            temporal_layer = Temporal_Sep_Layer
        elif layer_type == 'Bottelneck':
            temporal_layer = Temporal_Bottleneck_Layer
        elif layer_type == 'Basic':
            temporal_layer = Temporal_Basic_Layer
        else:
            raise ValueError("no such layer")

        for i, [channel, stride, depth] in enumerate(block_args):
            self.add_sublayer(
                f'block-{i}_scn',
                Spatial_Graph_Layer(last_channel, channel, max_graph_distance,
                                    **kwargs))
            for j in range(int(depth)):
                s = stride if j == 0 else 1
                self.add_sublayer(
                    f'block-{i}_tcn-{j}',
                    temporal_layer(
                        channel, temporal_window_size, stride=s, **kwargs))
            self.add_sublayer(f'block-{i}_att',
                              Attention_Layer(channel, **kwargs))
            last_channel = channel


class EfficientGCN_Classifier(nn.Sequential):
    def __init__(self, curr_channel, num_class, drop_prob, **kwargs):
        super(EfficientGCN_Classifier, self).__init__()

        self.add_sublayer('gap', nn.AdaptiveAvgPool3D(1))
        self.add_sublayer('dropout', nn.Dropout(drop_prob))
        self.add_sublayer('fc', nn.Conv3D(
            curr_channel, num_class, kernel_size=1))


def init_param(layers):
    for l in layers:
        if isinstance(l, nn.Conv1D) or isinstance(l, nn.Conv2D):
            weight_init_(l, 'KaimingNormal')
        elif isinstance(l, nn.BatchNorm1D) or isinstance(
                l, nn.BatchNorm2D) or isinstance(l, nn.BatchNorm3D):
            weight_init_(l, 'Constant', value=1)
            # l.weight.set_value(ones_(l.weight))
            # l.bias.set_value(zeros_(l.bias))
        elif isinstance(l, nn.Conv3D) or isinstance(l, nn.Linear):
            weight_init_(l, 'Normal')
            # l.weight.set_value(normal_(l.weight))
            # l.bias.set_value(zeros_(l.bias))
