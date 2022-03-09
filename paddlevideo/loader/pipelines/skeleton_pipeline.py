#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import os
import numpy as np
import paddle.nn.functional as F
import random
import paddle
from ..registry import PIPELINES
"""pipeline ops for Activity Net.
"""


@PIPELINES.register()
class AutoPadding(object):
    """
    Sample or Padding frame skeleton feature.
    Args:
        window_size: int, temporal size of skeleton feature.
        random_pad: bool, whether do random padding when frame length < window size. Default: False.
    """

    def __init__(self, window_size, random_pad=False):
        self.window_size = window_size
        self.random_pad = random_pad

    def get_frame_num(self, data):
        C, T, V, M = data.shape
        for i in range(T - 1, -1, -1):
            tmp = np.sum(data[:, i, :, :])
            if tmp > 0:
                T = i + 1
                break
        return T

    def __call__(self, results):
        data = results['data']

        C, T, V, M = data.shape
        T = self.get_frame_num(data)
        if T == self.window_size:
            data_pad = data[:, :self.window_size, :, :]
        elif T < self.window_size:
            begin = random.randint(0, self.window_size -
                                   T) if self.random_pad else 0
            data_pad = np.zeros((C, self.window_size, V, M))
            data_pad[:, begin:begin + T, :, :] = data[:, :T, :, :]
        else:
            if self.random_pad:
                index = np.random.choice(T, self.window_size,
                                         replace=False).astype('int64')
            else:
                index = np.linspace(0, T, self.window_size).astype("int64")
            data_pad = data[:, index, :, :]

        results['data'] = data_pad
        return results


@PIPELINES.register()
class SkeletonNorm(object):
    """
    Normalize skeleton feature.
    Args:
        aixs: dimensions of vertex coordinate. 2 for (x,y), 3 for (x,y,z). Default: 2.
    """

    def __init__(self, axis=2, squeeze=False):
        self.axis = axis
        self.squeeze = squeeze

    def __call__(self, results):
        data = results['data']

        # Centralization
        data = data - data[:, :, 8:9, :]
        data = data[:self.axis, :, :, :]  # get (x,y) from (x,y, acc)
        C, T, V, M = data.shape
        if self.squeeze:
            data = data.reshape((C, T, V))  # M = 1

        results['data'] = data.astype('float32')
        if 'label' in results:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results


@PIPELINES.register()
class Iden(object):
    """
    Wrapper Pipeline
    """

    def __init__(self, label_expand=True):
        self.label_expand = label_expand

    def __call__(self, results):
        data = results['data']
        results['data'] = data.astype('float32')

        if 'label' in results and self.label_expand:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results


@PIPELINES.register()
class RandomRotation(object):
    """
    Random rotation sketeton.
    Args:
        argument: bool, if rotation.
        theta: float, rotation rate.
    """

    def __init__(self, argument, theta=0.3):
        self.theta = theta
        self.argument = argument

    def _rot(self, rot):
        """
        rot: T,3
        """
        cos_r, sin_r = np.cos(rot), np.sin(rot)  # T,3
        zeros = np.zeros((rot.shape[0], 1))  # T,1
        ones = np.ones((rot.shape[0], 1))  # T,1

        r1 = np.stack((ones, zeros, zeros), axis=-1)  # T,1,3
        rx2 = np.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), axis=-1)  # T,1,3
        rx3 = np.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), axis=-1)  # T,1,3
        rx = np.concatenate((r1, rx2, rx3), axis=1)  # T,3,3

        ry1 = np.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), axis=-1)
        r2 = np.stack((zeros, ones, zeros), axis=-1)
        ry3 = np.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), axis=-1)
        ry = np.concatenate((ry1, r2, ry3), axis=1)

        rz1 = np.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), axis=-1)
        r3 = np.stack((zeros, zeros, ones), axis=-1)
        rz2 = np.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), axis=-1)
        rz = np.concatenate((rz1, rz2, r3), axis=1)

        rot = np.matmul(np.matmul(rz, ry), rx)
        return rot

    def __call__(self, results):
        # C,T,V,M
        data = results['data']
        if self.argument:
            C, T, V, M = data.shape
            data_numpy = np.transpose(data, (1, 0, 2, 3)).conjugate().reshape(
                T, C, V * M)  # T,3,V*M
            rot = np.random.uniform(-self.theta, self.theta, 3)
            rot = np.stack([
                rot,
            ] * T, axis=0)
            rot = self._rot(rot)  # T,3,3
            data_numpy = np.matmul(rot, data_numpy)
            data_numpy = data_numpy.reshape(T, C, V, M)
            data_numpy = np.transpose(data_numpy, (1, 0, 2, 3))
            data = data_numpy
        results['data'] = data.astype(np.float32)
        return results


@PIPELINES.register()
class SketeonCropSample(object):
    """
    Sketeon Crop Sampler.
    Args:
        crop_model: str, crop model, support: ['center'].
        p_interval: list, crop len
        window_size: int, sample windows size.
    """

    def __init__(self, window_size, crop_model='center', p_interval=1):
        assert crop_model in ['center'], "Don't support :" + crop_model

        self.crop_model = crop_model
        self.window_size = window_size
        self.p_interval = p_interval

    def __call__(self, results):
        if self.crop_model == 'center':
            # input: C,T,V,M
            data = results['data']
            valid_frame_num = np.sum(data.sum(0).sum(-1).sum(-1) != 0)

            C, T, V, M = data.shape
            begin = 0
            end = valid_frame_num
            valid_size = end - begin

            #crop
            if len(self.p_interval) == 1:
                p = self.p_interval[0]
                bias = int((1 - p) * valid_size / 2)
                data = data[:, begin + bias:end - bias, :, :]  # center_crop
                cropped_length = data.shape[1]
            else:
                p = np.random.rand(1) * (self.p_interval[1] - self.p_interval[0]
                                         ) + self.p_interval[0]
                # constraint cropped_length lower bound as 64
                cropped_length = np.minimum(
                    np.maximum(int(np.floor(valid_size * p)), 64), valid_size)
                bias = np.random.randint(0, valid_size - cropped_length + 1)
                data = data[:, begin + bias:begin + bias + cropped_length, :, :]

            # resize
            data = np.transpose(data, (0, 2, 3, 1)).conjugate().reshape(
                C * V * M, cropped_length)
            data = data[None, None, :, :]
            # could perform both up sample and down sample
            data_tensor = paddle.to_tensor(data)
            data_tensor = F.interpolate(data_tensor,
                                        size=(C * V * M, self.window_size),
                                        mode='bilinear',
                                        align_corners=False).squeeze()
            data = paddle.transpose(
                paddle.reshape(data_tensor, (C, V, M, self.window_size)),
                (0, 3, 1, 2)).numpy()
        else:
            raise NotImplementedError
        results['data'] = data
        return results


@PIPELINES.register()
class SketeonModalityTransform(object):
    """
    Sketeon Crop Sampler.
    Args:
        crop_model: str, crop model, support: ['center'].
        p_interval: list, crop len
        window_size: int, sample windows size.
    """

    def __init__(self, bone, motion, joint=True, graph='ntu_rgb_d'):

        self.joint = joint
        self.bone = bone
        self.motion = motion
        self.graph = graph
        if self.graph == "ntu_rgb_d":
            self.bone_pairs = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                               (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                               (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                               (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                               (22, 23), (21, 21), (23, 8), (24, 25), (25, 12))
        else:
            raise NotImplementedError

    def __call__(self, results):
        if self.joint:
            return results
        data_numpy = results['data']
        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in self.bone_pairs:
                bone_data_numpy[:, :, v1 -
                                1] = data_numpy[:, :, v1 -
                                                1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.motion:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        results['data'] = data_numpy
        return results
