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

import argparse
import cv2
import numpy as np
from PIL import Image
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from paddlevideo.loader.pipelines import Scale, CenterCrop, Normalization, Image2Array


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_file", type=str, help="video file path")
    parser.add_argument("--use_gpu", type=str2bool, default=True)

    # params for decode and sample
    parser.add_argument("--num_seg", type=int, default=8)
    parser.add_argument("--seg_len", type=int, default=1)

    # params for preprocess
    parser.add_argument("--short_size", type=int, default=256)
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--normalize", type=str2bool, default=True)

    # params for predict
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--params_file", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--enable_mkldnn", type=bool, default=False)
    parser.add_argument("--hubserving", type=str2bool, default=False)

    # params for infer

    parser.add_argument("--model", type=str)
    """
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--class_num", type=int, default=1000)
    parser.add_argument(
        "--load_static_weights",
        type=str2bool,
        default=False,
        help='Whether to load the pretrained weights saved in static mode')

    # parameters for pre-label the images
    parser.add_argument(
        "--pre_label_image",
        type=str2bool,
        default=False,
        help="Whether to pre-label the images using the loaded weights")
    parser.add_argument("--pre_label_out_idr", type=str, default=None)
    """

    return parser.parse_args()


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
    classes = np.argpartition(output, -args.top_k)[-args.top_k:]
    classes = classes[np.argsort(-output[classes])]
    scores = output[classes]
    return classes, scores
