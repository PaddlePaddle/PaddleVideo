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
import pickle

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import paddle
import paddle.nn.functional as F
import pandas
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from abc import abstractmethod

from paddlevideo.loader.pipelines import (
    AutoPadding, CenterCrop, DecodeSampler, FeatureDecoder, GroupResize,
    Image2Array, ImageDecoder, JitterScale, MultiCrop, Normalization,
    PackOutput, Sampler, Scale, SkeletonNorm, TenCrop, ToArray, UniformCrop,
    VideoDecoder)
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
        # file_info = json.load(open(input_file))
        self.feat_path = input_file
        self.video_duration = 12
        feat = np.load(input_file).astype('float32').T
        res = np.expand_dims(feat, axis=0).copy()

        return [res]

    def postprocess(self, outputs, outputs1, outputs2, outputs3, outputs4, outputs5, print_output=True):
        """
        output: list
        """
        pred_bm, pred_start, pred_end = outputs
        pred_bm1, pred_start1, pred_end1 = outputs1
        pred_bm2, pred_start2, pred_end2 = outputs2
        pred_bm3, pred_start3, pred_end3 = outputs3
        pred_bm4, pred_start4, pred_end4 = outputs4
        pred_bm5, pred_start5, pred_end5 = outputs5

        self._gen_props(pred_bm, pred_start[0], pred_end[0], pred_bm1, pred_start1[0], pred_end1[0], pred_bm2, pred_start2[0], pred_end2[0], pred_bm3, pred_start3[0], pred_end3[0], pred_bm4, pred_start4[0], pred_end4[0], pred_bm5, pred_start5[0], pred_end5[0], print_output)

    def _gen_props(self, pred_bm, pred_start, pred_end, pred_bm1, pred_start1, pred_end1, pred_bm2, pred_start2, pred_end2, pred_bm3, pred_start3, pred_end3, pred_bm4, pred_start4, pred_end4, pred_bm5, pred_start5, pred_end5, print_output):
        snippet_xmins = [1.0 / self.tscale * i for i in range(self.tscale)]
        snippet_xmaxs = [1.0 / self.tscale * i for i in range(1, self.tscale + 1)]

        pred_bm = pred_bm[0, 0, :, :] * pred_bm[0, 1, :, :]
        pred_bm1 = pred_bm1[0, 0, :, :] * pred_bm1[0, 1, :, :]
        pred_bm2 = pred_bm2[0, 0, :, :] * pred_bm2[0, 1, :, :]
        pred_bm3 = pred_bm3[0, 0, :, :] * pred_bm3[0, 1, :, :]
        pred_bm4 = pred_bm4[0, 0, :, :] * pred_bm4[0, 1, :, :]
        pred_bm5 = pred_bm5[0, 0, :, :] * pred_bm5[0, 1, :, :]

        start_mask = boundary_choose(pred_start)
        start_mask1 = boundary_choose(pred_start1)
        start_mask2 = boundary_choose(pred_start2)
        start_mask3 = boundary_choose(pred_start3)
        start_mask4 = boundary_choose(pred_start4)
        start_mask5 = boundary_choose(pred_start5)

        start_mask[0] = 1.
        start_mask1[0] = 1.
        start_mask2[0] = 1.
        start_mask3[0] = 1.
        start_mask4[0] = 1.
        start_mask5[0] = 1.

        end_mask = boundary_choose(pred_end)
        end_mask1 = boundary_choose(pred_end1)
        end_mask2 = boundary_choose(pred_end2)
        end_mask3 = boundary_choose(pred_end3)
        end_mask4 = boundary_choose(pred_end4)
        end_mask5 = boundary_choose(pred_end5)

        end_mask[-1] = 1.
        end_mask1[-1] = 1.
        end_mask2[-1] = 1.
        end_mask3[-1] = 1.
        end_mask4[-1] = 1.
        end_mask5[-1] = 1.

        score_vector_list = []
        for idx in range(self.dscale):
            for jdx in range(self.tscale):
                start_index = jdx
                end_index = start_index + idx
                if end_index < self.tscale and start_mask[start_index] == 1 and end_mask[end_index] == 1  and start_mask1[start_index] == 1 and end_mask1[end_index] == 1 and start_mask2[start_index] == 1 and end_mask2[end_index] == 1 and start_mask3[start_index] == 1 and end_mask3[end_index] == 1 and start_mask4[start_index] == 1 and end_mask4[end_index] == 1 and start_mask5[start_index] == 1 and end_mask5[end_index] == 1:
                    xmin = snippet_xmins[start_index]
                    xmax = snippet_xmaxs[end_index]

                    xmin_score = pred_start[start_index]
                    xmax_score = pred_end[end_index]

                    xmin_score1 = pred_start1[start_index]
                    xmax_score1 = pred_end1[end_index]

                    xmin_score2 = pred_start2[start_index]
                    xmax_score2 = pred_end2[end_index]

                    xmin_score3 = pred_start3[start_index]
                    xmax_score3 = pred_end3[end_index]

                    xmin_score4 = pred_start4[start_index]
                    xmax_score4 = pred_end4[end_index]

                    xmin_score5 = pred_start5[start_index]
                    xmax_score5 = pred_end5[end_index]

                    bm_score = pred_bm[idx, jdx]
                    bm_score1 = pred_bm1[idx, jdx]
                    bm_score2 = pred_bm2[idx, jdx]
                    bm_score3 = pred_bm3[idx, jdx]
                    bm_score4 = pred_bm4[idx, jdx]
                    bm_score5 = pred_bm5[idx, jdx]

                    conf_score = xmin_score * xmax_score * bm_score * xmin_score1 * xmax_score1 * bm_score1 * xmin_score2 * xmax_score2 * bm_score2 * xmin_score3 * xmax_score3 * bm_score3 * xmin_score4 * xmax_score4 * bm_score4 * xmin_score5 * xmax_score5 * bm_score5
                    
                    score_vector_list.append([xmin, xmax, conf_score])

        if score_vector_list == []:
            return 0
        else:
            cols = ["xmin", "xmax", "score"]
            score_vector_list = np.stack(score_vector_list)
            df = pandas.DataFrame(score_vector_list, columns=cols)

            result_dict = {}
            proposal_list = []
            df = soft_nms(df, alpha=0.4, t1=0.55, t2=0.9)
            for idx in range(min(60, len(df))):
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

            outfile = open(
                os.path.join(self.result_path, self.feat_path.split('.')[0] + ".json"), "w")

            json.dump(result_dict, outfile)

