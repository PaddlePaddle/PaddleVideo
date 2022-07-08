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
import warnings

import paddle
import paddle.nn as nn

from .resnet3d import ResNet3d, ConvBNLayer
from ..registry import BACKBONES


@BACKBONES.register()
class ResNet3dSlowOnly(ResNet3d):
    """A pathway of Slowfast based on ResNet3d.

    Args:
        *args (arguments): Arguments same as :class:``ResNet3d``.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to ``beta`` in the paper.
            Default: 8.
        **kwargs (keyword arguments): Keywords arguments for ResNet3d.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inplanes = self.base_channels

        self.lateral_connections = []
        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2**i
            self.inplanes = planes * self.block.expansion

    def make_res_layer(self,
                       block,
                       inplanes,
                       planes,
                       blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       inflate=1,
                       inflate_style='3x1x1',
                       non_local=0,
                       non_local_cfg=dict(),
                       conv_cfg=None,
                       norm_cfg=None,
                       act_cfg=None,
                       with_cp=False):
        """Build residual layer for Slowfast.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input
                feature in each block.
            planes (int): Number of channels for the output
                feature in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides
                in residual and conv layers. Default: 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Default: 1.
            dilation (int): Spacing between kernel elements. Default: 1.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Default: 1.
            inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines
                the kernel sizes and padding strides for conv1 and
                conv2 in each block. Default: ``3x1x1``.
            non_local (int | Sequence[int]): Determine whether to apply
                non-local module in the corresponding block of each stages.
                Default: 0.
            non_local_cfg (dict): Config for non-local module.
                Default: ``dict()``.
            conv_cfg (dict | None): Config for conv layers. Default: None.
            norm_cfg (dict | None): Config for norm layers. Default: None.
            act_cfg (dict | None): Config for activate layers. Default: None.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save
                some memory while slowing down the training speed.
                Default: False.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * blocks
        non_local = non_local if not isinstance(non_local,
                                                int) else (non_local, ) * blocks
        assert len(inflate) == blocks and len(non_local) == blocks

        lateral_inplanes = 0
        if (spatial_stride != 1
                or (inplanes + lateral_inplanes) != planes * block.expansion):
            downsample = ConvBNLayer(
                in_channels=inplanes + lateral_inplanes,
                out_channels=planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False,
                act=None)
        else:
            downsample = None

        layers = []
        layers.append(
            block(
                inplanes + lateral_inplanes,
                planes,
                spatial_stride,
                temporal_stride,
                dilation,
                downsample,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                non_local=(non_local[0] == 1),
                non_local_cfg=non_local_cfg,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp))
        inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    1,
                    1,
                    dilation,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    non_local=(non_local[i] == 1),
                    non_local_cfg=non_local_cfg,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp))

        return nn.Sequential(*layers)

    def _inflate_conv_params(self, conv3d, state_dict_2d, module_name_2d,
                             inflated_param_names):
        """Inflate a conv module from 2d to 3d.

        The differences of conv modules betweene 2d and 3d in Pathway
        mainly lie in the inplanes due to lateral connections. To fit the
        shapes of the lateral connection counterpart, it will expand
        parameters by concatting conv2d parameters and extra zero paddings.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        weight_2d_name = module_name_2d + '.weight'
        conv2d_weight = state_dict_2d[weight_2d_name]
        old_shape = conv2d_weight.shape
        new_shape = conv3d.weight.data.shape
        kernel_t = new_shape[2]

        if new_shape[1] != old_shape[1]:
            if new_shape[1] < old_shape[1]:
                warnings.warn(f'The parameter of {module_name_2d} is not'
                              'loaded due to incompatible shapes. ')
                return
            # Inplanes may be different due to lateral connections
            new_channels = new_shape[1] - old_shape[1]
            pad_shape = old_shape
            pad_shape = pad_shape[:1] + (new_channels, ) + pad_shape[2:]
            # Expand parameters by concat extra channels
            conv2d_weight = paddle.concat(
                (conv2d_weight, paddle.zeros(pad_shape)), axis=1)

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
            conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)


if __name__ == '__main__':
    net = ResNet3dSlowOnly(
        depth=50,
        in_channels=17,
        base_channels=32,
        conv1_kernel=(1, 7, 7),
        num_stages=3,
        out_indices=[2],
        stage_blocks=[3, 4, 6],
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=[0, 1, 1],
        with_pool2=False,
        spatial_strides=[2, 2, 2],
        temporal_strides=[1, 1, 2],
        dilations=[1, 1, 1])
    pass
