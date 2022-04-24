import collections.abc
from itertools import repeat
from typing import Any, Callable, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.layer import Identity

from ..registry import BACKBONES
from collections import OrderedDict

container_abcs = collections.abc
"""Model Config
"""

A0 = {'block_num': [0, 1, 3, 3, 4, 4]}
A0['conv1'] = [3, 8, (1, 3, 3), (1, 2, 2), (0, 1, 1)]
A0['b2_l0'] = [8, 8, 24, (1, 5, 5), (1, 2, 2), (0, 2, 2), (0, 1, 1)]
A0['b3_l0'] = [8, 32, 80, (3, 3, 3), (1, 2, 2), (1, 0, 0), (0, 0, 0)]
A0['b3_l1'] = [32, 32, 80, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)]
A0['b3_l2'] = [32, 32, 80, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)]
A0['b4_l0'] = [32, 56, 184, (5, 3, 3), (1, 2, 2), (2, 0, 0), (0, 0, 0)]
A0['b4_l1'] = [56, 56, 112, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)]
A0['b4_l2'] = [56, 56, 184, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)]
A0['b5_l0'] = [56, 56, 184, (5, 3, 3), (1, 1, 1), (2, 1, 1), (0, 1, 1)]
A0['b5_l1'] = [56, 56, 184, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)]
A0['b5_l2'] = [56, 56, 184, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)]
A0['b5_l3'] = [56, 56, 184, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)]
A0['b6_l0'] = [56, 104, 384, (5, 3, 3), (1, 2, 2), (2, 1, 1), (0, 1, 1)]
A0['b6_l1'] = [104, 104, 280, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1)]
A0['b6_l2'] = [104, 104, 280, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1)]
A0['b6_l3'] = [104, 104, 344, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1)]
A0['conv7'] = [104, 480, (1, 1, 1), (1, 1, 1), (0, 0, 0)]

MODEL_CONFIG = {'A0': A0}


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


def _make_divisible(v: float,
                    divisor: int,
                    min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8.
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class CausalModule(nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.activation = None

    def reset_activation(self) -> None:
        self.activation = None


class Conv2dBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Layer]] = None,
        activation_layer: Optional[Callable[..., nn.Layer]] = None,
        **kwargs: Any,
    ) -> None:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        if norm_layer is None:
            norm_layer = Identity
        if activation_layer is None:
            activation_layer = Identity
        self.kernel_size = kernel_size
        self.stride = stride
        dict_layers = (nn.Conv2D(in_planes,
                                 out_planes,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 groups=groups,
                                 **kwargs), norm_layer(out_planes,
                                                       momentum=0.1),
                       activation_layer())

        self.out_channels = out_planes
        super(Conv2dBNActivation, self).__init__(dict_layers[0], dict_layers[1],
                                                 dict_layers[2])


class Conv3DBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        padding: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Layer]] = None,
        activation_layer: Optional[Callable[..., nn.Layer]] = None,
        **kwargs: Any,
    ) -> None:
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        if norm_layer is None:
            norm_layer = Identity
        if activation_layer is None:
            activation_layer = Identity
        self.kernel_size = kernel_size
        self.stride = stride

        dict_layers = (nn.Conv3D(in_planes,
                                 out_planes,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 groups=groups,
                                 **kwargs), norm_layer(out_planes,
                                                       momentum=0.1),
                       activation_layer())
        self.out_channels = out_planes
        super(Conv3DBNActivation, self).__init__(dict_layers[0], dict_layers[1],
                                                 dict_layers[2])


class ConvBlock3D(CausalModule):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        causal: bool,
        conv_type: str,
        padding: Union[int, Tuple[int, int, int]] = 0,
        stride: Union[int, Tuple[int, int, int]] = 1,
        norm_layer: Optional[Callable[..., nn.Layer]] = None,
        activation_layer: Optional[Callable[..., nn.Layer]] = None,
        bias_attr: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        self.conv_2 = None

        if causal is True:
            padding = (0, padding[1], padding[2])
        if conv_type != "2plus1d" and conv_type != "3d":
            raise ValueError("only 2plus2d or 3d are " +
                             "allowed as 3d convolutions")

        if conv_type == "2plus1d":
            self.conv_1 = Conv2dBNActivation(in_planes,
                                             out_planes,
                                             kernel_size=(kernel_size[1],
                                                          kernel_size[2]),
                                             padding=(padding[1], padding[2]),
                                             stride=(stride[1], stride[2]),
                                             activation_layer=activation_layer,
                                             norm_layer=norm_layer,
                                             bias_attr=bias_attr,
                                             **kwargs)
            if kernel_size[0] > 1:
                self.conv_2 = Conv2dBNActivation(
                    in_planes,
                    out_planes,
                    kernel_size=(kernel_size[0], 1),
                    padding=(padding[0], 0),
                    stride=(stride[0], 1),
                    activation_layer=activation_layer,
                    norm_layer=norm_layer,
                    bias_attr=bias_attr,
                    **kwargs)
        elif conv_type == "3d":
            self.conv_1 = Conv3DBNActivation(in_planes,
                                             out_planes,
                                             kernel_size=kernel_size,
                                             padding=padding,
                                             activation_layer=activation_layer,
                                             norm_layer=norm_layer,
                                             stride=stride,
                                             bias_attr=bias_attr,
                                             **kwargs)
        self.padding = padding
        self.kernel_size = kernel_size
        self.dim_pad = self.kernel_size[0] - 1
        self.stride = stride
        self.causal = causal
        self.conv_type = conv_type

    def _forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.dim_pad > 0 and self.conv_2 is None and self.causal is True:
            x = self._cat_stream_buffer(x)
        b, c, t, h, w = x.shape
        if self.conv_type == "2plus1d":
            x = paddle.transpose(x, (0, 2, 1, 3, 4))  # bcthw --> btchw
            x = paddle.reshape_(x, (-1, c, h, w))  # btchw --> bt,c,h,w
        x = self.conv_1(x)
        if self.conv_type == "2plus1d":
            b, c, h, w = x.shape
            x = paddle.reshape_(x, (-1, t, c, h, w))  # bt,c,h,w --> b,t,c,h,w
            x = paddle.transpose(x, (0, 2, 1, 3, 4))  # b,t,c,h,w --> b,c,t,h,w
            if self.conv_2 is not None:
                if self.dim_pad > 0 and self.causal is True:
                    x = self._cat_stream_buffer(x)
                b, c, t, h, w = x.shape
                x = paddle.reshape_(x, (b, c, t, h * w))
                x = self.conv_2(x)
                b, c, t, _ = x.shape
                x = paddle.reshape_(x, (b, c, t, h, w))
        return x

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self._forward(x)
        return x

    def _cat_stream_buffer(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.activation is None:
            self._setup_activation(x.shape)
        x = paddle.concat((self.activation, x), 2)
        self._save_in_activation(x)
        return x

    def _save_in_activation(self, x: paddle.Tensor) -> None:
        assert self.dim_pad > 0
        self.activation = paddle.to_tensor(x.numpy()[:, :, -self.dim_pad:,
                                                     ...]).clone().detach()

    def _setup_activation(self, input_shape: Tuple[float, ...]) -> None:
        assert self.dim_pad > 0
        self.activation = paddle.zeros(shape=[
            *input_shape[:2],  # type: ignore
            self.dim_pad,
            *input_shape[3:]
        ])


class TemporalCGAvgPool3D(CausalModule):
    def __init__(self, ) -> None:
        super().__init__()
        self.n_cumulated_values = 0
        self.register_forward_post_hook(self._detach_activation)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        input_shape = x.shape
        cumulative_sum = paddle.cumsum(x, axis=2)
        if self.activation is None:
            self.activation = cumulative_sum[:, :, -1:].clone()
        else:
            cumulative_sum += self.activation
            self.activation = cumulative_sum[:, :, -1:].clone()

        noe = paddle.arange(1, input_shape[2] + 1)
        axis = paddle.to_tensor([0, 1, 3, 4])
        noe = paddle.unsqueeze(noe, axis=axis)
        divisor = noe.expand(x.shape)
        x = cumulative_sum / (self.n_cumulated_values + divisor)
        self.n_cumulated_values += input_shape[2]
        return x

    @staticmethod
    def _detach_activation(module: CausalModule, inputs: paddle.Tensor,
                           output: paddle.Tensor) -> None:
        module.activation.detach()

    def reset_activation(self) -> None:
        super().reset_activation()
        self.n_cumulated_values = 0


class SqueezeExcitation(nn.Layer):
    def __init__(self,
                 input_channels: int,
                 activation_2: nn.Layer,
                 activation_1: nn.Layer,
                 conv_type: str,
                 causal: bool,
                 squeeze_factor: int = 4,
                 bias_attr: bool = True) -> None:
        super().__init__()
        self.causal = causal
        se_multiplier = 2 if causal else 1
        squeeze_channels = _make_divisible(
            input_channels // squeeze_factor * se_multiplier, 8)
        self.temporal_cumualtive_GAvg3D = TemporalCGAvgPool3D()
        self.fc1 = ConvBlock3D(input_channels * se_multiplier,
                               squeeze_channels,
                               kernel_size=(1, 1, 1),
                               padding=0,
                               causal=causal,
                               conv_type=conv_type,
                               bias_attr=bias_attr)
        self.activation_1 = activation_1()
        self.activation_2 = activation_2()
        self.fc2 = ConvBlock3D(squeeze_channels,
                               input_channels,
                               kernel_size=(1, 1, 1),
                               padding=0,
                               causal=causal,
                               conv_type=conv_type,
                               bias_attr=bias_attr)

    def _scale(self, inputs: paddle.Tensor) -> paddle.Tensor:
        if self.causal:
            x_space = paddle.mean(inputs, axis=[3, 4], keepdim=True)
            scale = self.temporal_cumualtive_GAvg3D(x_space)
            scale = paddle.concat((scale, x_space), axis=1)
        else:
            scale = F.adaptive_avg_pool3d(inputs, 1)
        scale = self.fc1(scale)
        scale = self.activation_1(scale)
        scale = self.fc2(scale)
        return self.activation_2(scale)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        scale = self._scale(inputs)
        return scale * inputs


class BasicBneck(nn.Layer):
    def __init__(
        self,
        input_channels,
        out_channels,
        expanded_channels,
        kernel_size,
        stride,
        padding,
        padding_avg,
        causal: bool,
        conv_type: str,
        norm_layer: Optional[Callable[..., nn.Layer]] = None,
        activation_layer: Optional[Callable[..., nn.Layer]] = None,
    ) -> None:
        super().__init__()

        assert type(stride) is tuple

        if (not stride[0] == 1 or not (1 <= stride[1] <= 2)
                or not (1 <= stride[2] <= 2)):
            raise ValueError('illegal stride value')

        self.res = None

        layers = []
        if expanded_channels != out_channels:
            # expand
            self.expand = ConvBlock3D(in_planes=input_channels,
                                      out_planes=expanded_channels,
                                      kernel_size=(1, 1, 1),
                                      padding=(0, 0, 0),
                                      causal=causal,
                                      conv_type=conv_type,
                                      norm_layer=norm_layer,
                                      activation_layer=activation_layer)
        # deepwise
        self.deep = ConvBlock3D(in_planes=expanded_channels,
                                out_planes=expanded_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                groups=expanded_channels,
                                causal=causal,
                                conv_type=conv_type,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)

        # SE
        self.se = SqueezeExcitation(
            expanded_channels,
            causal=causal,
            activation_1=activation_layer,
            activation_2=(nn.Sigmoid if conv_type == "3d" else nn.Hardsigmoid),
            conv_type=conv_type)
        # project
        self.project = ConvBlock3D(expanded_channels,
                                   out_channels,
                                   kernel_size=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   causal=causal,
                                   conv_type=conv_type,
                                   norm_layer=norm_layer,
                                   activation_layer=Identity)

        if not (stride == (1, 1, 1) and input_channels == out_channels):
            if stride != (1, 1, 1):
                layers.append(
                    nn.AvgPool3D((1, 3, 3), stride=stride, padding=padding_avg))
            layers.append(
                ConvBlock3D(
                    in_planes=input_channels,
                    out_planes=out_channels,
                    kernel_size=(1, 1, 1),
                    padding=(0, 0, 0),
                    norm_layer=norm_layer,
                    activation_layer=Identity,
                    causal=causal,
                    conv_type=conv_type,
                ))
            self.res = nn.Sequential(*layers)
        self.alpha = self.create_parameter(shape=[1], dtype="float32")

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        if self.res is not None:
            residual = self.res(inputs)
        else:
            residual = inputs
        if self.expand is not None:
            x = self.expand(inputs)
        else:
            x = inputs

        x = self.deep(x)
        x = self.se(x)
        x = self.project(x)
        result = residual + self.alpha * x
        return result


@BACKBONES.register()
class MoViNet(nn.Layer):
    def __init__(
        self,
        model_type: str = 'A0',
        hidden_dim: int = 2048,
        causal: bool = True,
        num_classes: int = 400,
        conv_type: str = "3d",
    ) -> None:
        super().__init__()
        """
        causal: causal mode
        num_classes: number of classes for classifcation
        conv_type: type of convolution either 3d or 2plus1d
        """
        blocks_dic = OrderedDict()
        cfg = MODEL_CONFIG[model_type]

        norm_layer = nn.BatchNorm3D if conv_type == "3d" else nn.BatchNorm2D
        activation_layer = nn.Swish if conv_type == "3d" else nn.Hardswish

        # conv1
        self.conv1 = ConvBlock3D(in_planes=cfg['conv1'][0],
                                 out_planes=cfg['conv1'][1],
                                 kernel_size=cfg['conv1'][2],
                                 stride=cfg['conv1'][3],
                                 padding=cfg['conv1'][4],
                                 causal=causal,
                                 conv_type=conv_type,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer)
        # blocks
        for i in range(2, len(cfg['block_num']) + 1):
            for j in range(cfg['block_num'][i - 1]):
                blocks_dic[f'b{i}_l{j}'] = BasicBneck(
                    cfg[f'b{i}_l{j}'][0],
                    cfg[f'b{i}_l{j}'][1],
                    cfg[f'b{i}_l{j}'][2],
                    cfg[f'b{i}_l{j}'][3],
                    cfg[f'b{i}_l{j}'][4],
                    cfg[f'b{i}_l{j}'][5],
                    cfg[f'b{i}_l{j}'][6],
                    causal=causal,
                    conv_type=conv_type,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer)
        self.blocks = nn.Sequential(*(blocks_dic.values()))

        # conv7
        self.conv7 = ConvBlock3D(in_planes=cfg['conv7'][0],
                                 out_planes=cfg['conv7'][1],
                                 kernel_size=cfg['conv7'][2],
                                 stride=cfg['conv7'][3],
                                 padding=cfg['conv7'][4],
                                 causal=causal,
                                 conv_type=conv_type,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer)
        # pool
        self.classifier = nn.Sequential(
            # dense9
            ConvBlock3D(in_planes=cfg['conv7'][1],
                        out_planes=hidden_dim,
                        kernel_size=(1, 1, 1),
                        causal=causal,
                        conv_type=conv_type,
                        bias_attr=True),
            nn.Swish(),
            nn.Dropout(p=0.2),
            # dense10d
            ConvBlock3D(in_planes=hidden_dim,
                        out_planes=num_classes,
                        kernel_size=(1, 1, 1),
                        causal=causal,
                        conv_type=conv_type,
                        bias_attr=True),
        )
        if causal:
            self.cgap = TemporalCGAvgPool3D()
        self.apply(self._weight_init)
        self.causal = causal

    def avg(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.causal:
            avg = F.adaptive_avg_pool3d(x, (x.shape[2], 1, 1))
            avg = self.cgap(avg)[:, :, -1:]
        else:
            avg = F.adaptive_avg_pool3d(x, 1)
        return avg

    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Conv3D):
            nn.initializer.KaimingNormal(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(0.0)(m.bias)
        elif isinstance(m, (nn.BatchNorm3D, nn.BatchNorm2D, nn.GroupNorm)):
            nn.initializer.Constant(1.0)(m.weight)
            nn.initializer.Constant(0.0)(m.bias)
        elif isinstance(m, nn.Linear):
            nn.initializer.Normal(m.weight, 0, 0.01)
            nn.initializer.Constant(0.0)(m.bias)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv7(x)
        x = self.avg(x)
        x = self.classifier(x)
        x = x.flatten(1)
        return x

    @staticmethod
    def _clean_activation_buffers(m):
        if issubclass(type(m), CausalModule):
            m.reset_activation()

    def clean_activation_buffers(self) -> None:
        self.apply(self._clean_activation_buffers)


if __name__ == '__main__':
    net = MoViNet(causal=False, conv_type='3d')
    paddle.summary(net, input_size=(1, 3, 8, 224, 224))
