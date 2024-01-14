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

from ..registry import BACKBONES
from .darknet import Darknet
from .resnext101 import ResNext101
import paddle.nn as nn
import paddle


class CAM_Module(nn.Layer):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        temp = paddle.zeros([1], dtype='float32')
        self.gamma = paddle.create_parameter(shape=temp.shape, dtype=str(temp.numpy().dtype),
                                             default_initializer=paddle.nn.initializer.Assign(temp))
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.shape
        proj_query = paddle.reshape(x, [m_batchsize, C, -1])
        proj_key = paddle.transpose(paddle.reshape(
            x, [m_batchsize, C, -1]), perm=[0, 2, 1])
        energy = paddle.bmm(proj_query, proj_key)
        energy_new = paddle.expand_as(paddle.max(
            energy, axis=-1, keepdim=True), energy) - energy
        attention = self.softmax(energy_new)
        proj_value = paddle.reshape(x, [m_batchsize, C, -1])

        out = paddle.bmm(attention, proj_value)
        out = out.reshape([m_batchsize, C, height, width])
        out = self.gamma * out + x
        return out


class CFAMBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(CFAMBlock, self).__init__()
        inter_channels = 1024
        self.conv_bn_relu1 = nn.Sequential(nn.Conv2D(in_channels, inter_channels, kernel_size=1, bias_attr=False),
                                           nn.BatchNorm2D(inter_channels),
                                           nn.ReLU())
        self.conv_bn_relu2 = nn.Sequential(nn.Conv2D(inter_channels, inter_channels, 3, padding=1, bias_attr=False),
                                           nn.BatchNorm2D(inter_channels),
                                           nn.ReLU())

        self.sc = CAM_Module(inter_channels)

        self.conv_bn_relu3 = nn.Sequential(nn.Conv2D(inter_channels, inter_channels, 3, padding=1, bias_attr=False),
                                           nn.BatchNorm2D(inter_channels),
                                           nn.ReLU())
        self.conv_out = nn.Sequential(nn.Dropout2D(0.1), nn.Conv2D(
            inter_channels, out_channels, 1, bias_attr=True))

    def forward(self, x):
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.sc(x)
        x = self.conv_bn_relu3(x)
        output = self.conv_out(x)

        return output


@BACKBONES.register()
class YOWO(nn.Layer):
    def __init__(self, num_class, pretrained_2d=None, pretrained_3d=None):
        super(YOWO, self).__init__()

        self.pretrained_2d = pretrained_2d
        self.pretrained_3d = pretrained_3d
        self.backbone_2d = Darknet()
        self.backbone_3d = ResNext101()
        self.num_ch_2d = 425
        self.num_ch_3d = 2048
        self.num_class = num_class
        self.cfam = CFAMBlock(self.num_ch_2d + self.num_ch_3d, 1024)
        self.conv_final = nn.Conv2D(
            1024, 5 * (self.num_class + 4 + 1), kernel_size=1, bias_attr=False)
        self.seen = 0

    def init_weights(self):
        if self.pretrained_2d is not None:
            self.backbone_2d = self.load_pretrain_weight(
                self.backbone_2d, self.pretrained_2d)
        if self.pretrained_3d is not None:
            self.backbone_3d = self.load_pretrain_weight(
                self.backbone_3d, self.pretrained_3d)

    def load_pretrain_weight(self, model, weights_path):
        model_dict = model.state_dict()

        param_state_dict = paddle.load(weights_path)
        ignore_weights = set()

        # hack: fit for faster rcnn. Pretrain weights contain prefix of 'backbone'
        # while res5 module is located in bbox_head.head. Replace the prefix of
        # res5 with 'bbox_head.head' to load pretrain weights correctly.
        for k in list(param_state_dict.keys()):
            if 'backbone.res5' in k:
                new_k = k.replace('backbone', 'bbox_head.head')
                if new_k in model_dict.keys():
                    value = param_state_dict.pop(k)
                    param_state_dict[new_k] = value

        for name, weight in param_state_dict.items():
            if name in model_dict.keys():
                if list(weight.shape) != list(model_dict[name].shape):
                    print(
                        '{} not used, shape {} unmatched with {} in model.'.format(
                            name, weight.shape, list(model_dict[name].shape)))
                    ignore_weights.add(name)
            else:
                print('Redundant weight {} and ignore it.'.format(name))
                ignore_weights.add(name)

        for weight in ignore_weights:
            param_state_dict.pop(weight, None)

        model.set_dict(param_state_dict)
        print('Finish loading model weights: {}'.format(weights_path))
        return model

    def forward(self, input):
        x_3d = input  # Input clip
        x_2d = input[:, :, -1, :, :]  # Last frame of the clip that is read

        x_2d = self.backbone_2d(x_2d)

        x_3d = self.backbone_3d(x_3d)

        x_3d = paddle.squeeze(x_3d, axis=2)

        x = paddle.concat([x_3d, x_2d], axis=1)
        x = self.cfam(x)
        out = self.conv_final(x)

        return out
