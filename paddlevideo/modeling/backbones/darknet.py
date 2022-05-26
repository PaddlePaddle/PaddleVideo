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
from .yolo_cfg import parse_cfg, print_cfg, load_conv_bn, load_conv, load_fc, save_conv_bn, save_conv, save_fc


class MaxPoolStride1(nn.Layer):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        pad = nn.Pad2D(padding=[0, 1, 0, 1], mode='replicate')
        x = F.max_pool2d(pad(x), 2, stride=1)
        return x


class Reorg(nn.Layer):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.dim() == 4)
        B = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.reshape([B, C, H // hs, hs, W // ws, ws]).transpose([0, 1, 2, 4, 3, 5])
        x = x.reshape([B, C, H // hs * W // ws, hs * ws]).transpose([0, 1, 3, 2])
        x = x.reshape([B, C, hs * ws, H // hs, W // ws]).transpose([0, 2, 1, 3, 4])
        x = x.reshape([B, hs * ws * C, H // hs, W // ws])
        return x


class GlobalAvgPool2d(nn.Layer):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        x = F.avg_pool2d(x, (H, W))
        x = x.reshape([N, C])
        return x


class EmptyModule(nn.Layer):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


class Darknet(nn.Layer):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.loss = None
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks)  # merge conv, bn,leaky

        self.header = paddle.to_tensor([0, 0, 0, 0]).astype('int32')
        self.seen = 0

    def forward(self, x):
        ind = -2
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1

            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'reorg' or block[
                'type'] == 'avgpool' or block['type'] == 'softmax' or block['type'] == 'connected':
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = paddle.concat([x1, x2], 1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1)
                elif activation == 'relu':
                    x = F.relu(x)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        return x

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.LayerList()

        prev_filters = 3
        out_filters = []
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_sublayer('conv',
                                       nn.Conv2D(prev_filters, filters, kernel_size, stride, pad, bias_attr=False))
                    model.add_sublayer('bn', nn.BatchNorm2D(filters))
                    # model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_sublayer('conv', nn.Conv2D(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_sublayer('leaky', nn.LeakyReLU(0.1))
                elif activation == 'relu':
                    model.add_sublayer('relu', nn.ReLU())
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2D(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss()
                elif block['_type'] == 'L1':
                    model = nn.L1Loss()
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss()
                out_filters.append(1)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert (layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters, bias_attr=True)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters, bias_attr=True),
                        nn.LeakyReLU(0.1))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters, bias_attr=True),
                        nn.ReLU())
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'region':
                pass  # DO NOTHING
            else:
                print('unknown type %s' % (block['type']))

        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = paddle.to_tensor(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()
        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model['conv'], model['bn'])
                else:
                    start = load_conv(buf, start, model['conv'])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
