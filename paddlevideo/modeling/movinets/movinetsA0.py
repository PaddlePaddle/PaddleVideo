import collections.abc
from collections import OrderedDict
from itertools import repeat
from typing import Any, Callable, Optional, Tuple, Union
from typing import List

import paddle
from einops import rearrange
from paddle import nn

container_abcs = collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))


def _list_with_default(out_size, defaults):
    # type: (List[int], List[int]) -> List[int]
    if isinstance(out_size, int):
        return out_size
    if len(defaults) <= len(out_size):
        raise ValueError(
            'Input dimension should be at least {}'.format(len(out_size) + 1))
    return [
        v if v is not None else d
        for v, d in zip(out_size, defaults[-len(out_size):])
    ]


def _make_divisible(v: float, divisor: int,
                    min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
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


def same_padding(x: paddle.Tensor, in_height: int, in_width: int, stride_h: int,
                 stride_w: int, filter_height: int,
                 filter_width: int) -> paddle.Tensor:
    if in_height % stride_h == 0:
        pad_along_height = max(filter_height - stride_h, 0)
    else:
        pad_along_height = max(filter_height - (in_height % stride_h), 0)
    if in_width % stride_w == 0:
        pad_along_width = max(filter_width - stride_w, 0)
    else:
        pad_along_width = max(filter_width - (in_width % stride_w), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding_pad = (pad_left, pad_right, pad_top, pad_bottom)
    return paddle.nn.functional.pad(x, padding_pad)


class Identity(nn.Layer):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)
    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        return inputs


class CausalModule(nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.activation = None

    def reset_activation(self) -> None:
        self.activation = None


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


class Conv2dBNActivation(nn.Sequential):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            *,
            kernel_size: Union[int, Tuple[int, int]],
            padding: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            norm_layer: Optional[Callable[..., nn.Layer]] = None,
            activation_layer: Optional[Callable[..., paddle.nn.Layer]] = None,
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
            *,
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

        dict_layers = (
            nn.Conv3D(in_planes,
                      out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      **kwargs),
            norm_layer(out_planes, momentum=0.1),  # eps=0.001),
            activation_layer())
        self.out_channels = out_planes
        super(Conv3DBNActivation, self).__init__(dict_layers[0], dict_layers[1],
                                                 dict_layers[2])


class ConvBlock3D(CausalModule):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            *,
            kernel_size: Union[int, Tuple[int, int, int]],
            tf_like: bool,
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
        if tf_like:
            # We neek odd kernel to have even padding
            # and stride == 1 to precompute padding,
            if kernel_size[0] % 2 == 0:
                raise ValueError('tf_like supports only odd' +
                                 ' kernels for temporal dimension')
            padding = ((kernel_size[0] - 1) // 2, 0, 0)
            if stride[0] != 1:
                raise ValueError('illegal stride value, tf like supports' +
                                 ' only stride == 1 for temporal dimension')
            if stride[1] > kernel_size[1] or stride[2] > kernel_size[2]:
                # these values are not tested so should be avoided
                raise ValueError('tf_like supports only' +
                                 '  stride <= of the kernel size')

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
        self.tf_like = tf_like

    def _forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.dim_pad > 0 and self.conv_2 is None and self.causal is True:
            x = self._cat_stream_buffer(x)
        shape_with_buffer = x.shape
        if self.conv_type == "2plus1d":
            x = x.numpy()
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = paddle.to_tensor(x)
        x = self.conv_1(x)
        if self.conv_type == "2plus1d":
            x = x.numpy()
            x = rearrange(x, "(b t) c h w -> b c t h w", t=shape_with_buffer[2])
            x = paddle.to_tensor(x)
            if self.conv_2 is not None:
                if self.dim_pad > 0 and self.causal is True:
                    x = self._cat_stream_buffer(x)
                w = x.shape[-1]
                x = x.numpy()
                x = rearrange(x, "b c t h w -> b c t (h w)")
                x = paddle.to_tensor(x)
                x = self.conv_2(x)
                x = x.numpy()
                x = rearrange(x, "b c t (h w) -> b c t h w", w=w)
                x = paddle.to_tensor(x)
        return x

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.tf_like:
            x = same_padding(x, x.shape[-2], x.shape[-1], self.stride[-2],
                             self.stride[-1], self.kernel_size[-2],
                             self.kernel_size[-1])
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
        self.activation = paddle.to_tensor(
            x.numpy()[:, :, -self.dim_pad:, ...]).clone().detach()

    def _setup_activation(self, input_shape: Tuple[float, ...]) -> None:
        assert self.dim_pad > 0
        self.activation = paddle.zeros(shape=[
            *input_shape[:2],  # type: ignore
            self.dim_pad,
            *input_shape[3:]
        ])


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
                               tf_like=False,
                               causal=causal,
                               conv_type=conv_type,
                               bias_attr=bias_attr)
        self.activation_1 = activation_1()
        self.activation_2 = activation_2()
        self.fc2 = ConvBlock3D(squeeze_channels,
                               input_channels,
                               kernel_size=(1, 1, 1),
                               padding=0,
                               tf_like=False,
                               causal=causal,
                               conv_type=conv_type,
                               bias_attr=bias_attr)

    def _scale(self, inputs: paddle.Tensor) -> paddle.Tensor:
        if self.causal:
            x_space = paddle.mean(inputs, axis=[3, 4], keepdim=True)
            scale = self.temporal_cumualtive_GAvg3D(x_space)
            scale = paddle.concat((scale, x_space), axis=1)
        else:
            scale = paddle.nn.functional.adaptive_avg_pool3d(inputs, 1)
        scale = self.fc1(scale)
        scale = self.activation_1(scale)
        scale = self.fc2(scale)
        return self.activation_2(scale)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        scale = self._scale(inputs)
        return scale * inputs


class tfAvgPool3D(nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.avgf = paddle.nn.AvgPool3D((1, 3, 3), stride=(1, 2, 2))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if x.shape[-1] != x.shape[-2]:
            raise RuntimeError('only same shape for h and w ' +
                               'are supported by avg with tf_like')
        if x.shape[-1] != x.shape[-2]:
            raise RuntimeError('only same shape for h and w ' +
                               'are supported by avg with tf_like')
        f1 = x.shape[-1] % 2 != 0
        if f1:
            padding_pad = (0, 0, 0, 0)
        else:
            padding_pad = (0, 1, 0, 1)
        x = paddle.nn.functional.pad(x, padding_pad)
        if f1:
            x = paddle.nn.functional.avg_pool3d(x, (1, 3, 3),
                                                stride=(1, 2, 2),
                                                padding=(0, 1, 1))
        else:
            x = self.avgf(x)
            x[..., -1] = x[..., -1] * 9 / 6
            x[..., -1, :] = x[..., -1, :] * 9 / 6
        return x


class BasicBneck(nn.Layer):
    def __init__(
            self,
            cfg,
            causal: bool,
            tf_like: bool,
            conv_type: str,
            norm_layer: Optional[Callable[..., nn.Layer]] = None,
            activation_layer: Optional[Callable[..., nn.Layer]] = None,
    ) -> None:
        super().__init__()
        assert type(cfg["stride"]) is tuple
        if (not cfg["stride"][0] == 1 or not (1 <= cfg["stride"][1] <= 2)
                or not (1 <= cfg["stride"][2] <= 2)):
            raise ValueError('illegal stride value')
        self.res = None

        layers = []
        if cfg["expanded_channels"] != cfg["out_channels"]:
            # expand
            self.expand = ConvBlock3D(in_planes=cfg["input_channels"],
                                      out_planes=cfg["expanded_channels"],
                                      kernel_size=(1, 1, 1),
                                      padding=(0, 0, 0),
                                      causal=causal,
                                      conv_type=conv_type,
                                      tf_like=tf_like,
                                      norm_layer=norm_layer,
                                      activation_layer=activation_layer)
        # deepwise
        self.deep = ConvBlock3D(in_planes=cfg["expanded_channels"],
                                out_planes=cfg["expanded_channels"],
                                kernel_size=cfg["kernel_size"],
                                padding=cfg["padding"],
                                stride=cfg["stride"],
                                groups=cfg["expanded_channels"],
                                causal=causal,
                                conv_type=conv_type,
                                tf_like=tf_like,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)
        # SE
        self.se = SqueezeExcitation(
            cfg["expanded_channels"],
            causal=causal,
            activation_1=activation_layer,
            activation_2=(nn.Sigmoid if conv_type == "3d" else nn.Hardsigmoid),
            conv_type=conv_type)
        # project
        self.project = ConvBlock3D(cfg["expanded_channels"],
                                   cfg["out_channels"],
                                   kernel_size=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   causal=causal,
                                   conv_type=conv_type,
                                   tf_like=tf_like,
                                   norm_layer=norm_layer,
                                   activation_layer=Identity)

        if not (cfg["stride"] == (1, 1, 1)
                and cfg["input_channels"] == cfg["out_channels"]):
            if cfg["stride"] != (1, 1, 1):
                if tf_like:
                    layers.append(tfAvgPool3D())
                else:
                    layers.append(
                        paddle.nn.AvgPool3D((1, 3, 3),
                                            stride=cfg["stride"],
                                            padding=cfg["padding_avg"]))
            layers.append(
                ConvBlock3D(in_planes=cfg["input_channels"],
                            out_planes=cfg["out_channels"],
                            kernel_size=(1, 1, 1),
                            padding=(0, 0, 0),
                            norm_layer=norm_layer,
                            activation_layer=Identity,
                            causal=causal,
                            conv_type=conv_type,
                            tf_like=tf_like))
            self.res = nn.Sequential(*layers)
        self.alpha = paddle.static.create_parameter(shape=[1], dtype="float32")

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


def _no_grad_zero_(tensor):
    with paddle.no_grad():
        return paddle.zeros(tensor.shape)


def _no_grad_fill_ones(tensor):
    with paddle.no_grad():
        return paddle.ones(shape=tensor.shape)


class MoViNet(nn.Layer):
    def __init__(
            self,
            # state_dict_path,
            causal: bool = True,
            pretrained: bool = False,
            num_classes: int = 600,
            conv_type: str = "3d",
            tf_like: bool = False):
        """
        调用MoViNetA0
        Args:
            causal:是否调用因果卷积
            pretrained: 是否调用预训练模型
            num_classes: 类别数
            conv_type: 卷积核类型
            tf_like: 参考为tensorflow的类型
        """
        super().__init__()
        if pretrained:
            tf_like = True
            num_classes = 600
            conv_type = "2plus1d" if causal else "3d"
        blocks_dic = OrderedDict()
        norm_layer = nn.BatchNorm3D if conv_type == "3d" else nn.BatchNorm2D
        activation_layer = paddle.nn.Swish if conv_type == "3d" else paddle.nn.Hardswish
        # conv1
        self.conv1 = ConvBlock3D(in_planes=3,
                                 out_planes=8,
                                 kernel_size=(1, 3, 3),
                                 stride=(1, 2, 2),
                                 padding=(0, 1, 1),
                                 causal=causal,
                                 conv_type=conv_type,
                                 tf_like=tf_like,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer)
        # blocks
        blocks_dic['b0_l0'] = BasicBneck(cfg={
            "input_channels": 8,
            "out_channels": 8,
            "expanded_channels": 24,
            "kernel_size": (1, 5, 5),
            "stride": (1, 2, 2),
            "padding": (0, 2, 2),
            "padding_avg": (0, 1, 1)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        blocks_dic['b1_l0'] = BasicBneck(cfg={
            "input_channels": 8,
            "out_channels": 32,
            "expanded_channels": 80,
            "kernel_size": (3, 3, 3),
            "stride": (1, 2, 2),
            "padding": (1, 0, 0),
            "padding_avg": (0, 0, 0)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        blocks_dic['b1_l1'] = BasicBneck(cfg={
            "input_channels": 32,
            "out_channels": 32,
            "expanded_channels": 80,
            "kernel_size": (3, 3, 3),
            "stride": (1, 1, 1),
            "padding": (1, 1, 1),
            "padding_avg": (0, 1, 1)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        blocks_dic['b1_l2'] = BasicBneck(cfg={
            "input_channels": 32,
            "out_channels": 32,
            "expanded_channels": 80,
            "kernel_size": (3, 3, 3),
            "stride": (1, 1, 1),
            "padding": (1, 1, 1),
            "padding_avg": (0, 1, 1)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        blocks_dic['b2_l0'] = BasicBneck(cfg={
            "input_channels": 32,
            "out_channels": 56,
            "expanded_channels": 184,
            "kernel_size": (5, 3, 3),
            "stride": (1, 2, 2),
            "padding": (2, 0, 0),
            "padding_avg": (0, 0, 0)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        blocks_dic['b2_l1'] = BasicBneck(cfg={
            "input_channels": 56,
            "out_channels": 56,
            "expanded_channels": 112,
            "kernel_size": (3, 3, 3),
            "stride": (1, 1, 1),
            "padding": (1, 1, 1),
            "padding_avg": (0, 1, 1)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        blocks_dic['b2_l2'] = BasicBneck(cfg={
            "input_channels": 56,
            "out_channels": 56,
            "expanded_channels": 184,
            "kernel_size": (3, 3, 3),
            "stride": (1, 1, 1),
            "padding": (1, 1, 1),
            "padding_avg": (0, 1, 1)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        blocks_dic['b3_l0'] = BasicBneck(cfg={
            "input_channels": 56,
            "out_channels": 56,
            "expanded_channels": 184,
            "kernel_size": (5, 3, 3),
            "stride": (1, 1, 1),
            "padding": (2, 1, 1),
            "padding_avg": (0, 1, 1)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        blocks_dic['b3_l1'] = BasicBneck(cfg={
            "input_channels": 56,
            "out_channels": 56,
            "expanded_channels": 184,
            "kernel_size": (5, 3, 3),
            "stride": (1, 1, 1),
            "padding": (1, 1, 1),
            "padding_avg": (0, 1, 1)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        blocks_dic['b3_l2'] = BasicBneck(cfg={
            "input_channels": 56,
            "out_channels": 56,
            "expanded_channels": 184,
            "kernel_size": (5, 3, 3),
            "stride": (1, 1, 1),
            "padding": (1, 1, 1),
            "padding_avg": (0, 1, 1)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        blocks_dic['b3_l3'] = BasicBneck(cfg={
            "input_channels": 56,
            "out_channels": 56,
            "expanded_channels": 184,
            "kernel_size": (5, 3, 3),
            "stride": (1, 1, 1),
            "padding": (1, 1, 1),
            "padding_avg": (0, 1, 1)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        blocks_dic['b4_l0'] = BasicBneck(cfg={
            "input_channels": 56,
            "out_channels": 104,
            "expanded_channels": 384,
            "kernel_size": (5, 3, 3),
            "stride": (1, 1, 1),
            "padding": (1, 1, 1),
            "padding_avg": (0, 1, 1)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        blocks_dic['b4_l1'] = BasicBneck(cfg={
            "input_channels": 104,
            "out_channels": 104,
            "expanded_channels": 280,
            "kernel_size": (1, 5, 5),
            "stride": (1, 1, 1),
            "padding": (0, 2, 2),
            "padding_avg": (0, 1, 1)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        blocks_dic['b4_l2'] = BasicBneck(cfg={
            "input_channels": 104,
            "out_channels": 104,
            "expanded_channels": 280,
            "kernel_size": (1, 5, 5),
            "stride": (1, 1, 1),
            "padding": (0, 2, 2),
            "padding_avg": (0, 1, 1)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        blocks_dic['b4_l3'] = BasicBneck(cfg={
            "input_channels": 104,
            "out_channels": 104,
            "expanded_channels": 344,
            "kernel_size": (1, 5, 5),
            "stride": (1, 1, 1),
            "padding": (0, 2, 2),
            "padding_avg": (0, 1, 1)
        },
                                         causal=causal,
                                         conv_type=conv_type,
                                         tf_like=tf_like,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)
        self.blocks = nn.Sequential(
            blocks_dic['b0_l0'], blocks_dic['b1_l0'], blocks_dic['b1_l1'],
            blocks_dic['b1_l2'], blocks_dic['b2_l0'], blocks_dic['b2_l1'],
            blocks_dic['b2_l2'], blocks_dic['b3_l0'], blocks_dic['b3_l1'],
            blocks_dic['b3_l2'], blocks_dic['b3_l3'], blocks_dic['b4_l0'],
            blocks_dic['b4_l1'], blocks_dic['b4_l2'], blocks_dic['b4_l3'])
        # conv7
        self.conv7 = ConvBlock3D(in_planes=104,
                                 out_planes=480,
                                 kernel_size=(1, 1, 1),
                                 stride=(1, 1, 1),
                                 padding=(0, 0, 0),
                                 causal=causal,
                                 conv_type=conv_type,
                                 tf_like=tf_like,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer)
        # pool
        self.classifier = nn.Sequential(
            # dense9
            ConvBlock3D(480,
                        2048,
                        kernel_size=(1, 1, 1),
                        tf_like=tf_like,
                        causal=causal,
                        conv_type=conv_type,
                        bias_attr=False),
            nn.Swish(),
            # nn.Dropout(p=0.2, inplace=True),
            nn.Dropout(p=0.2, ),
            # dense10d
            ConvBlock3D(2048,
                        num_classes,
                        kernel_size=(1, 1, 1),
                        tf_like=tf_like,
                        causal=causal,
                        conv_type=conv_type,
                        bias_attr=True),
        )
        if causal:
            self.cgap = TemporalCGAvgPool3D()
        else:
            self.apply(self._weight_init)
        self.causal = causal

    def avg(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.causal:
            avg = paddle.nn.functional.adaptive_avg_pool3d(
                x, (x.shape[2], 1, 1))
            avg = self.cgap(avg)[:, :, -1:]
        else:
            avg = paddle.nn.functional.adaptive_avg_pool3d(x, 1)
        return avg

    @staticmethod
    def _weight_init(m):  # TODO check this

        if isinstance(m, nn.Conv3D):
            nn.initializer.KaimingNormal(m.weight)
            if m.bias is not None:
                _no_grad_zero_(m.bias)  # maybe have problems
        elif isinstance(m, (nn.BatchNorm3D, nn.BatchNorm2D, nn.GroupNorm)):
            _no_grad_fill_ones(m.weight)
            _no_grad_zero_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.initializer.Normal(m.weight, 0, 0.01)
            _no_grad_zero_(m.bias)

    def _forward_impl(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv7(x)
        x = self.avg(x)
        x = self.classifier(x)
        x = x.flatten(1)

        return x

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self._forward_impl(x)

    @staticmethod
    def _clean_activation_buffers(m):
        if issubclass(type(m), CausalModule):
            m.reset_activation()

    def clean_activation_buffers(self) -> None:
        self.apply(self._clean_activation_buffers)


if __name__ == '__main__':
    paddle.set_device("cpu")
    model = MoViNet(num_classes=6, causal=True)
    # 1 为batch 3为通道数 12为帧长 172*172是视频文件每一帧的像素值
    input_shape = [1, 3, 12, 172, 172]
    test_data = paddle.ones(shape=input_shape)
    test_out = model(test_data)
