from math import sqrt
from typing import Callable, Tuple

import paddle
import paddle.nn as nn
from paddle.nn import functional as F
from paddle.nn.initializer import Constant, Normal, Uniform

from ...utils import load_ckpt
from ..registry import BACKBONES
from ..weight_init import _calculate_fan_in_and_fan_out, kaiming_uniform_

__all__ = ['WAFPNet']

zeros_ = Constant(value=0.)


class Conv_ReLU_Block(nn.Layer):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2D(in_channels=64,
                              out_channels=64,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias_attr=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class Attention_net(nn.Layer):
    def __init__(self):
        super(Attention_net, self).__init__()
        self.Kc = paddle.nn.Conv2D(in_channels=64,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias_attr=False)
        self.Qc = paddle.nn.Conv2D(in_channels=64,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias_attr=False)
        self.Vc = paddle.nn.Conv2D(in_channels=64,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias_attr=False)
        self.local_weight = paddle.nn.Conv2D(in_channels=64,
                                             out_channels=64,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             bias_attr=False)

        self.a = self.create_parameter(shape=(1, ), default_initializer=zeros_)
        self.add_parameter("a", self.a)

        self.b = self.create_parameter(shape=(1, ), default_initializer=zeros_)
        self.add_parameter("b", self.b)

    def forward(self, x):
        kc = self.Kc(x)
        vc = self.Vc(x)
        qc = self.Qc(x)

        numel_hw = x.shape[-2] * x.shape[-1]
        vc_reshape = vc.reshape([-1, x.shape[1], numel_hw])
        vc_reshape = vc_reshape.transpose([0, 2, 1])
        qc_reshape = qc.reshape([-1, x.shape[1], numel_hw])
        kc_reshape = kc.reshape([-1, x.shape[1], numel_hw])
        kc_reshape = kc_reshape.transpose([0, 2, 1])

        qvc = paddle.matmul(qc_reshape, vc_reshape)
        attentionc = F.softmax(qvc, axis=-1)

        vector_c = paddle.matmul(kc_reshape, attentionc)
        vector_c_reshape = vector_c.transpose([0, 2, 1])

        Oc = vector_c_reshape.reshape(
            [x.shape[0], x.shape[1], x.shape[2], x.shape[3]])

        vp_reshape = vc.reshape([-1, x.shape[1], numel_hw])
        vp_reshape = vp_reshape.transpose([0, 2, 1])
        qp_reshape = qc.reshape([-1, x.shape[1], numel_hw])
        kp_reshape = kc.reshape([-1, x.shape[1], numel_hw])
        kp_reshape = kp_reshape.transpose([0, 2, 1])

        kqp = paddle.matmul(kp_reshape, qp_reshape)
        attention_p = F.softmax(kqp, axis=-1)

        vector_p = paddle.matmul(attention_p, vp_reshape)
        vector_p_reshape = vector_p.transpose([0, 2, 1])
        Op = vector_p_reshape.reshape(
            [x.shape[0], x.shape[1], x.shape[2], x.shape[3]])

        # O = self.a * Oc + self.b * Op + x
        O = paddle.add(
            paddle.add(paddle.multiply(self.a, Oc), paddle.multiply(self.b,
                                                                    Op)), x)

        out = self.local_weight(O)
        return out


@BACKBONES.register()
class WAFPNet(nn.Layer):
    def __init__(self,
                 pretrained: str = None,
                 num_residual_layer: int = 9,
                 num_refine_layer: int = 9):
        super(WAFPNet, self).__init__()
        self.pretrained = pretrained

        self.input1 = nn.Conv2D(in_channels=1,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias_attr=False)
        self.input5 = nn.Conv2D(in_channels=3,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias_attr=False)
        self.relu = nn.ReLU()
        self.attention = Attention_net()
        self.output = nn.Conv2D(in_channels=64,
                                out_channels=1,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias_attr=False)

        self.residual_layer = self.make_layer(Conv_ReLU_Block,
                                              num_residual_layer)
        self.refine_layer = self.make_layer(Conv_ReLU_Block, num_refine_layer)

    def make_layer(self, block: Callable, num_of_layer: int):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def init_weights(self):
        """First init model's weight"""
        for m in self.sublayers(True):
            # init Conv layers.
            if isinstance(m, nn.Conv2D):
                kaiming_uniform_(m.weight, a=sqrt(5))
                if m.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / sqrt(fan_in)
                    uniform_ = Uniform(-bound, bound)
                    uniform_(m.bias)
            # init Linear layers.
            elif isinstance(m, nn.Linear):
                kaiming_uniform_(m.weight, a=sqrt(5))
                if m.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
                    uniform_ = Uniform(-bound, bound)
                    uniform_(m.bias)

        for m in self.sublayers(True):
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[-1] * m.weight.shape[-2] * m.weight.shape[0]
                normal_ = Normal(0, sqrt(2.0 / n))
                normal_(m.weight)
        """Second, if provide pretrained ckpt, load it"""
        if isinstance(
                self.pretrained, str
        ) and self.pretrained.strip() != "":  # load pretrained weights
            load_ckpt(self, self.pretrained)

    def forward(
        self, x: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        residual = x
        out = self.relu(self.input1(x))
        out = self.residual_layer(out)
        out = self.attention(out)
        out = self.output(out)
        out = paddle.add(out, residual)
        residual1 = out

        out = self.relu(self.input1(out))
        out = self.residual_layer(out)
        out = self.attention(out)
        out = self.output(out)
        out = paddle.add(out, residual1)
        residual2 = out

        out = self.relu(self.input1(out))
        out = self.residual_layer(out)
        out = self.attention(out)
        out = self.output(out)
        out = paddle.add(out, residual2)
        residual3 = out
        resi2 = paddle.concat((residual1, residual2), 1)
        resi3 = paddle.concat((residual3, resi2), 1)

        out = self.relu(self.input5(resi3))
        out = self.refine_layer(out)
        out = self.attention(out)
        out = self.output(out)
        out = paddle.add(out, x)

        return residual1, residual2, residual3, out
