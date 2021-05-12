# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import numpy as np
import sys
import paddle.nn.functional as F
import paddle
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from paddlevideo.loader.pipelines import VideoDecoder, Sampler, Scale, CenterCrop, Normalization, Image2Array
from paddlevideo.utils import build, Registry

INFERENCE = Registry('inference')


def build_inference_helper(cfg):
    return build(cfg, INFERENCE)


@INFERENCE.register()
class ppTSM_Inference_helper():
    def __init__(self,
                 num_seg=8,
                 seg_len=1,
                 short_size=256,
                 target_size=224,
                 top_k=1):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.short_size = short_size
        self.target_size = target_size
        self.top_k = top_k

    def preprocess(self, input_file):
        """
        input_file: str, file path
        input_flie_list: str, file list path.
        """
        self.input_file = input_file
        assert os.path.isfile(input_file) is not None, "{} not exists".format(
            input_file)
        results = {'filename': input_file}
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        ops = [
            VideoDecoder(),
            Sampler(self.num_seg, self.seg_len, valid_mode=True),
            Scale(self.short_size),
            CenterCrop(self.target_size),
            Image2Array(),
            Normalization(img_mean, img_std)
        ]
        for op in ops:
            results = op(results)

        res = np.expand_dims(results['imgs'], axis=0).copy()
        return [res]

    def postprocess(self, output):
        """
        output: list
        """
        output = output[0].flatten()
        output = F.softmax(paddle.to_tensor(output)).numpy()
        classes = np.argpartition(output, -self.top_k)[-self.top_k:]
        classes = classes[np.argsort(-output[classes])]
        scores = output[classes]
        print("Current video file: {}".format(self.input_file))
        print("\ttop-1 class: {0}".format(classes[0]))
        print("\ttop-1 score: {0}".format(scores[0]))
