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

import json
import os
import sys

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
import pandas
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from paddlevideo.loader.pipelines import (AutoPadding, CenterCrop,
                                          DecodeSampler, FeatureDecoder,
                                          Image2Array, JitterScale, MultiCrop,
                                          Normalization, PackOutput, Sampler,
                                          Scale, SkeletonNorm, TenCrop,
                                          UniformCrop, VideoDecoder)
from paddlevideo.metrics.bmn_metric import boundary_choose, soft_nms
from paddlevideo.utils import Registry, build

INFERENCE = Registry('inference')


def decode(filepath, args):
    num_seg = args.num_seg
    seg_len = args.seg_len

    cap = cv2.VideoCapture(filepath)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampledFrames = []
    for i in range(videolen):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        sampledFrames.append(img)
    average_dur = int(len(sampledFrames) / num_seg)
    imgs = []
    for i in range(num_seg):
        idx = 0
        if average_dur >= seg_len:
            idx = (average_dur - 1) // 2
            idx += i * average_dur
        elif average_dur >= 1:
            idx += i * average_dur
        else:
            idx = i

        for jj in range(idx, idx + seg_len):
            imgbuf = sampledFrames[int(jj % len(sampledFrames))]
            img = Image.fromarray(imgbuf, mode='RGB')
            imgs.append(img)

    return imgs


def preprocess(img, args):
    img = {"imgs": img}
    resize_op = Scale(short_size=args.short_size)
    img = resize_op(img)
    ccrop_op = CenterCrop(target_size=args.target_size)
    img = ccrop_op(img)
    to_array = Image2Array()
    img = to_array(img)
    if args.normalize:
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        normalize_op = Normalization(mean=img_mean, std=img_std)
        img = normalize_op(img)
    return img['imgs']


def postprocess(output, args):
    output = output.flatten()
    output = F.softmax(paddle.to_tensor(output)).numpy()
    classes = np.argpartition(output, -args.top_k)[-args.top_k:]
    classes = classes[np.argsort(-output[classes])]
    scores = output[classes]
    return classes, scores


def build_inference_helper(cfg):
    return build(cfg, INFERENCE)


class Base_Inference_helper():
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
        raise NotImplementedError

    def postprocess(self, output, print_output=True, top_k=1):
        """
        output: list
        """
        if not isinstance(self.input_file, list):
            self.input_file = [
                self.input_file,
            ]
        output = output[0]  # [B, num_cls]
        N = output.shape[0]
        output = F.softmax(paddle.to_tensor(output), axis=-1).numpy()
        for i in range(N):
            classes = np.argpartition(output[i], -self.top_k)[-self.top_k:]
            classes = classes[np.argsort(-output[i, classes])]
            scores = output[i, classes]
            if print_output:
                print("Current video file: {0}".format(self.input_file[i]))
                for j in range(self.top_k):
                    print("\ttop-{0} class: {1}".format(j + 1, classes[j]))
                    print("\ttop-{0} score: {1}".format(j + 1, scores[j]))


@INFERENCE.register()
class ppTSM_Inference_helper(Base_Inference_helper):
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
        return: list
        """
        self.input_file = [input_file]
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
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


@INFERENCE.register()
class ppTSN_Inference_helper(Base_Inference_helper):
    def __init__(self,
                 num_seg=25,
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
        return: list
        """
        self.input_file = [input_file]
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        results = {'filename': input_file}
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        ops = [
            VideoDecoder(),
            Sampler(self.num_seg,
                    self.seg_len,
                    valid_mode=True,
                    select_left=True),
            Scale(self.short_size,
                  fixed_ratio=True,
                  do_round=True,
                  backend='cv2'),
            TenCrop(self.target_size),
            Image2Array(),
            Normalization(img_mean, img_std)
        ]
        for op in ops:
            results = op(results)

        res = np.expand_dims(results['imgs'], axis=0).copy()
        return [res]


@INFERENCE.register()
class BMN_Inference_helper(Base_Inference_helper):
    def __init__(self, feat_dim, dscale, tscale, result_path):
        self.feat_dim = feat_dim
        self.dscale = dscale
        self.tscale = tscale
        self.result_path = result_path
        if not os.path.isdir(self.result_path):
            os.makedirs(self.result_path)

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        file_info = json.load(open(input_file))
        self.feat_path = file_info['feat_path']
        self.video_duration = file_info['duration_second']
        feat = np.load(self.feat_path).astype('float32').T
        res = np.expand_dims(feat, axis=0).copy()

        return [res]

    def postprocess(self, outputs):
        """
        output: list
        """
        pred_bm, pred_start, pred_end = outputs
        self._gen_props(pred_bm, pred_start[0], pred_end[0])

    def _gen_props(self, pred_bm, pred_start, pred_end, print_output=True):
        snippet_xmins = [1.0 / self.tscale * i for i in range(self.tscale)]
        snippet_xmaxs = [
            1.0 / self.tscale * i for i in range(1, self.tscale + 1)
        ]

        pred_bm = pred_bm[0, 0, :, :] * pred_bm[0, 1, :, :]
        start_mask = boundary_choose(pred_start)
        start_mask[0] = 1.
        end_mask = boundary_choose(pred_end)
        end_mask[-1] = 1.
        score_vector_list = []
        for idx in range(self.dscale):
            for jdx in range(self.tscale):
                start_index = jdx
                end_index = start_index + idx
                if end_index < self.tscale and start_mask[
                        start_index] == 1 and end_mask[end_index] == 1:
                    xmin = snippet_xmins[start_index]
                    xmax = snippet_xmaxs[end_index]
                    xmin_score = pred_start[start_index]
                    xmax_score = pred_end[end_index]
                    bm_score = pred_bm[idx, jdx]
                    conf_score = xmin_score * xmax_score * bm_score
                    score_vector_list.append([xmin, xmax, conf_score])

        cols = ["xmin", "xmax", "score"]
        score_vector_list = np.stack(score_vector_list)
        df = pandas.DataFrame(score_vector_list, columns=cols)

        result_dict = {}
        proposal_list = []
        df = soft_nms(df, alpha=0.4, t1=0.55, t2=0.9)
        for idx in range(min(100, len(df))):
            tmp_prop={"score":df.score.values[idx], \
                      "segment":[max(0,df.xmin.values[idx])*self.video_duration, \
                                 min(1,df.xmax.values[idx])*self.video_duration]}
            proposal_list.append(tmp_prop)

        result_dict[self.feat_path] = proposal_list

        # print top-5 predictions
        if print_output:
            print("Current video file: {0} :".format(self.feat_path))
            for pred in proposal_list[:5]:
                print(pred)

        # save result
        outfile = open(
            os.path.join(self.result_path, "bmn_results_inference.json"), "w")

        json.dump(result_dict, outfile)


@INFERENCE.register()
class TimeSformer_Inference_helper(Base_Inference_helper):
    def __init__(self,
                 num_seg=8,
                 seg_len=1,
                 short_size=224,
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
        return: list
        """
        self.input_file = [input_file]
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        results = {'filename': input_file}
        img_mean = [0.45, 0.45, 0.45]
        img_std = [0.225, 0.225, 0.225]
        ops = [
            VideoDecoder(backend='pyav', mode='test', num_seg=self.num_seg),
            Sampler(self.num_seg,
                    self.seg_len,
                    valid_mode=True,
                    linspace_sample=True),
            Normalization(img_mean, img_std, tensor_shape=[1, 1, 1, 3]),
            Image2Array(data_format='cthw'),
            JitterScale(self.short_size, self.short_size),
            UniformCrop(self.target_size)
        ]
        for op in ops:
            results = op(results)

        # [N,C,Tx3,H,W]
        res = np.expand_dims(results['imgs'], axis=0).copy()
        return [res]


@INFERENCE.register()
class SlowFast_Inference_helper(Base_Inference_helper):
    def __init__(self,
                 num_frames=32,
                 sampling_rate=2,
                 target_size=256,
                 alpha=8,
                 top_k=1):
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.target_size = target_size
        self.alpha = alpha
        self.top_k = top_k

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        self.input_file = [input_file]
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        results = {
            'filename': input_file,
            'temporal_sample_index': 0,
            'spatial_sample_index': 0,
            'temporal_num_clips': 1,
            'spatial_num_clips': 1
        }
        img_mean = [0.45, 0.45, 0.45]
        img_std = [0.225, 0.225, 0.225]
        ops = [
            DecodeSampler(self.num_frames, self.sampling_rate, test_mode=True),
            JitterScale(self.target_size, self.target_size),
            MultiCrop(self.target_size),
            Image2Array(transpose=False),
            Normalization(img_mean, img_std, tensor_shape=[1, 1, 1, 3]),
            PackOutput(self.alpha),
        ]
        for op in ops:
            results = op(results)

        res = []
        for item in results['imgs']:
            res.append(np.expand_dims(item, axis=0).copy())
        return res

    def postprocess(self, output, print_output=True):
        """
        output: list
        """
        if not isinstance(self.input_file, list):
            self.input_file = [
                self.input_file,
            ]
        output = output[0]  # [B, num_cls]
        # output = F.softmax(paddle.to_tensor(output), axis=-1).numpy() # done in it's head
        N = output.shape[0]
        for i in range(N):
            classes = np.argpartition(output[i], -self.top_k)[-self.top_k:]
            classes = classes[np.argsort(-output[i, classes])]
            scores = output[i, classes]
            if print_output:
                print("Current video file: {0}".format(self.input_file[i]))
                for j in range(self.top_k):
                    print("\ttop-{0} class: {1}".format(j + 1, classes[j]))
                    print("\ttop-{0} score: {1}".format(j + 1, scores[j]))


@INFERENCE.register()
class STGCN_Inference_helper(Base_Inference_helper):
    def __init__(self,
                 num_channels,
                 window_size,
                 vertex_nums,
                 person_nums,
                 top_k=1):
        self.num_channels = num_channels
        self.window_size = window_size
        self.vertex_nums = vertex_nums
        self.person_nums = person_nums
        self.top_k = top_k

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        self.input_file = [input_file]
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        data = np.load(self.input_file)
        results = {'data': data}
        ops = [AutoPadding(window_size=self.window_size), SkeletonNorm()]
        for op in ops:
            results = op(results)

        res = np.expand_dims(results['data'], axis=0).copy()
        return [res]


@INFERENCE.register()
class AttentionLSTM_Inference_helper(Base_Inference_helper):
    def __init__(
            self,
            num_classes,  #Optional, the number of classes to be classified.
            feature_num,
            feature_dims,
            embedding_size,
            lstm_size,
            top_k=1):
        self.num_classes = num_classes
        self.feature_num = feature_num
        self.feature_dims = feature_dims
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.top_k = top_k

    def preprocess(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        self.input_file = [input_file]
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)
        results = {'filename': input_file}
        ops = [FeatureDecoder(num_classes=self.num_classes, has_label=False)]
        for op in ops:
            results = op(results)

        res = []
        for modality in ['rgb', 'audio']:
            res.append(
                np.expand_dims(results[f'{modality}_data'], axis=0).copy())
            res.append(
                np.expand_dims(results[f'{modality}_len'], axis=0).copy())
            res.append(
                np.expand_dims(results[f'{modality}_mask'], axis=0).copy())
        return res
