import paddle
import paddlenlp
from paddle import nn
import paddle.nn.functional as F

class Basic_Layer(nn.Layer):
    def __init__(self, in_channel, out_channel, residual, bias, act, **kwargs):
        super(Basic_Layer, self).__init__()

        self.conv = nn.Conv2D(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), bias_attr=bias)
        self.bn = nn.BatchNorm2D(out_channel)

        self.residual = nn.Identity() if residual else Zero_Layer()
        self.act = act

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.bn(self.conv(x)) + res)
        return x

class Spatial_Graph_Layer(Basic_Layer):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, residual=True, **kwargs):
        super(Spatial_Graph_Layer, self).__init__(in_channel, out_channel, residual, bias, **kwargs)

        self.conv = SpatialGraphConv(in_channel, out_channel,  max_graph_distance, bias, **kwargs)

        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2D(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, 1), stride=[1, 1], bias_attr=bias),
                nn.BatchNorm2D(out_channel),
            )

class Temporal_Basic_Layer(Basic_Layer):
    def __init__(self, channel, temporal_window_size, bias, stride=1, residual=True, **kwargs):
        super(Temporal_Basic_Layer, self).__init__(channel, channel, residual, bias, **kwargs)

        padding = (temporal_window_size - 1) // 2
        self.conv = nn.Conv2D(in_channels=channel, out_channels=channel, kernel_size=(temporal_window_size, 1), stride=(stride, 1), padding=(padding, 0), bias_attr=bias)
        if residual and stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2D(in_channels=channel, out_channels=channel, kernel_size=1, stride=(stride, 1), bias_attr=bias),
                nn.BatchNorm2D(channel)
            )


class Temporal_Bottleneck_Layer(nn.Layer):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_Bottleneck_Layer, self).__init__()
        inner_channel = channel // reduct_ratio
        padding = (temporal_window_size - 1) // 2
        self.act = act
        self.reduct_conv = nn.Sequential(
            nn.Conv2D(in_channels=channel, out_channels=inner_channel, kernel_size=1, stride=(1, 1),bias_attr=bias),
            nn.BatchNorm2D(inner_channel),
        )

        self.conv = nn.Sequential(
            nn.Conv2D(in_channels=inner_channel, out_channels=inner_channel, kernel_size=(temporal_window_size, 1), stride=(stride, 1), padding=(padding, 0), bias_attr=bias),
            nn.BatchNorm2D(inner_channel),
        )
        self.expand_conv = nn.Sequential(
            nn.Conv2D(in_channels=inner_channel, out_channels=channel, kernel_size=1, stride=(1, 1), bias_attr=bias),
            nn.BatchNorm2D(channel),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2D(in_channels=channel, out_channels=channel, kernel_size=1, stride=(stride, 1), bias_attr=bias),
                nn.BatchNorm2D(channel)
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.reduct_conv(x))
        x = self.act(self.conv(x))
        x = self.act(self.expand_conv(x) + res)
        return x

class Temporal_Sep_Layer(nn.Layer):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_Sep_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        self.act = act

        if expand_ratio > 0:
            inner_channel = channel * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2D(in_channels=channel, out_channels=inner_channel, kernel_size=1, stride=(1, 1), bias_attr=bias),
                nn.BatchNorm2D(inner_channel),
            )
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2D(in_channels=inner_channel, out_channels=inner_channel, kernel_size=(temporal_window_size, 1), stride=(stride, 1), padding=(padding, 1), groups=inner_channel, bias_attr=bias, weight_attr=nn.initializer.KaimingNormal()),
            nn.BatchNorm2D(inner_channel),
        )

        self.point_conv = nn.Sequential(
            nn.Conv2D(in_channels=inner_channel, out_channels=channel, kernel_size=1, stride=(1, 1), bias_attr=bias),
            nn.BatchNorm2D(channel),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2D(in_channels=channel, out_channels=channel, kernel_size=1, stride=(stride, 1), bias_attr=bias),
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
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_SG_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        inner_channel = channel // reduct_ratio
        self.act = act

        self.depth_conv1 = nn.Sequential(
            nn.Conv2D(in_channels=channel, out_channels=channel, kernel_size=(temporal_window_size, 1), stride=(1, 1)
            , padding=(padding, 0), groups=channel, bias_attr=bias),
            nn.BatchNorm2D(channel),
        )

        self.point_conv1 = nn.Sequential(
            nn.Conv2D(in_channels=channel, out_channels=inner_channel, kernel_size=1, stride=(1, 1), bias_attr=bias),
            nn.BatchNorm2D(inner_channel),
        )

        self.point_conv2 = nn.Sequential(
            nn.Conv2D(in_channels=inner_channel, out_channels=channel, kernel_size=1, stride=(1, 1), bias_attr=bias),
            nn.BatchNorm2D(channel),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2D(in_channels=channel, out_channels=channel, kernel_size=(temporal_window_size, 1)
            , stride=(stride, 1), padding=(padding, 0), groups=channel, bias_attr=bias),
            nn.BatchNorm2D(channel),
        )
        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2D(in_channels=channel, out_channels=channel, kernel_size=1, stride=(stride, 1), bias_attr=bias),
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
    def __init__(self, in_channel, out_channel,  max_graph_distance, bias, edge, A, **kwargs):
        super(SpatialGraphConv, self).__init__()

        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2D(in_channel, out_channel * self.s_kernel_size, 1, stride=(1, 1), bias_attr=bias)
        A = A[:self.s_kernel_size]
        _A = self.create_parameter(
            shape=A.shape, dtype='float32', attr=paddle.ParamAttr(initializer=nn.initializer.Assign(A), trainable=False)
        )
        self.add_parameter('A', _A)

        if edge:
            ones = paddle.ones_like(self.A)
            _edge = self.create_parameter(
                shape=ones.shape, dtype='float32', attr=paddle.ParamAttr(initializer=nn.initializer.Assign(ones), trainable=True)
            )
            self.add_parameter('edge', _edge)
        else:
            _edge = self.create_parameter(
                shape=[1], dtype='float32', attr=paddle.ParamAttr(initializer=nn.initializer.Assign(paddle.to_tensor(1)), trainable=True)
            )
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















