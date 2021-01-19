#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import ParamAttr

from ..model import ModelBase

__all__ = ["AudioNet"]


class AudioNet(ModelBase):
    """AudioNet"""
    def __init__(self, name, cfg, mode='train'):
        super(AudioNet, self).__init__(name, cfg, mode=mode)
        self.cfg = cfg
        self.name = name
        self.mode = mode
        self.py_reader = None
        self.get_config()

    def get_config(self):
        """get_config"""
        # get model configs
        self.feature_names = self.cfg[self.name.upper()]['feature_names']
        self.feature_dims = self.cfg[self.name.upper()]['feature_dims']

    def build_input(self, use_dataloader):
        """build_input"""
        self.feature_input = []
        for name, dim in zip(self.feature_names, self.feature_dims):
            # print("*******", name, dim)
            self.feature_input.append(
                fluid.layers.data(shape=dim, dtype='float32', name=name))
        #self.feature_input.append(
        #    fluid.layers.data(
        #        shape=[96, 64], lod_level=1, dtype='float32', name='audio'))
        if self.mode != 'infer':
            self.label_input = fluid.layers.data(shape=[1],
                                                 dtype='int64',
                                                 name='label')
        else:
            self.label_input = None
        if use_dataloader:
            assert self.mode != 'infer', \
                'pyreader is not recommendated when infer, please set use_pyreader to be false.'
            self.py_reader = fluid.io.PyReader(feed_list=self.feature_input +
                                               [self.label_input],
                                               capacity=1024,
                                               iterable=True)

    def conv_block(self, input, num_filter, groups, name=None):
        """conv_block """
        conv = input
        for i in range(groups):
            conv = fluid.layers.conv2d(input=conv,
                                       num_filters=num_filter,
                                       filter_size=3,
                                       stride=1,
                                       padding=1,
                                       act='relu')
        return conv

    def conv_block_1conv(self, input, num_filter, groups, name=None):
        """conv_block """
        conv = input
        for i in range(groups):
            conv = fluid.layers.conv2d(input=conv,
                                       num_filters=num_filter,
                                       filter_size=1,
                                       stride=1,
                                       act='relu')
        return conv

    def build_model(self, layers=6):
        #print("audio.shape", self.feature_input[0].shape)
        vgg_spec = {
            6: ([1, 1, 2, 2]),
            11: ([1, 1, 2, 2, 2]),
            13: ([2, 2, 2, 2, 2]),
            16: ([2, 2, 3, 3, 3]),
            19: ([2, 2, 4, 4, 4])
        }
        nums = vgg_spec[layers]
        input = fluid.layers.unsqueeze(input=self.feature_input[0], axes=[1])
        # print("input.shape", input.shape)

        conv1 = self.conv_block(input, 64, nums[0])
        pool1 = fluid.layers.pool2d(input=conv1,
                                    pool_size=2,
                                    pool_type='max',
                                    pool_stride=2)
        conv2 = self.conv_block(pool1, 128, nums[1])
        pool2 = fluid.layers.pool2d(input=conv2,
                                    pool_size=2,
                                    pool_type='max',
                                    pool_stride=2)
        conv3 = self.conv_block(pool2, 256, nums[2])
        pool3 = fluid.layers.pool2d(input=conv3,
                                    pool_size=2,
                                    pool_type='max',
                                    pool_stride=2)
        conv4 = self.conv_block(pool3, 512, nums[3])  # (-1, 512, 6, 8)
        # print("********conv4.shape = ", conv4.shape)
        pool4 = fluid.layers.pool2d(input=conv4,
                                    pool_size=2,
                                    pool_type='max',
                                    pool_stride=2)
        # print("********pool4.shape = ", pool4.shape) #(-1, 512, 3, 4)
        conv5 = self.conv_block(pool4, 1024, 1)  # (-1, 1024, 3, 4)
        # print("********conv5.shape = ", conv5.shape)
        pool5 = fluid.layers.pool2d(input=conv5,
                                    pool_type='avg',
                                    global_pooling=True)  # (-1, 1024, 1, 1)
        # print("********pool5.shape = ", pool5.shape)
        pool_reshape = fluid.layers.reshape(pool5, shape=[-1, 1024, 1, 1])
        pool_out = fluid.layers.squeeze(input=pool_reshape, axes=[2, 3])
        # print("********pool_out.shape = ", pool_out.shape)

        self.network_outputs = [pool_out]

    def outputs(self):
        """outputs"""
        return self.network_outputs

    def feeds(self):
        """feeds"""
        return self.feature_input if self.mode == 'infer' else self.feature_input + [
            self.label_input
        ]

    def fetches(self):
        """fetches"""
        if self.mode == 'train' or self.mode == 'valid':
            losses = self.loss()
            fetch_list = [losses, self.network_outputs[0], self.label_input]
        elif self.mode == 'infer':
            #fetch_list = [self.network_outputs[0], self.network_outputs[1]]
            fetch_list = [self.network_outputs[0]]
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

        return fetch_list
