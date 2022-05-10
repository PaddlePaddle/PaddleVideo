# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import sys

import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../../")))

from paddle_serving_client import Client
from preprocess_ops import Compose
from paddlevideo.loader.pipelines import (CenterCrop, Image2Array,
                                          Normalization, Sampler, Scale,
                                          VideoDecoder)
from typing import Any, Dict, Tuple, List


def preprocess(video_path: str) -> Tuple[Dict[str, np.ndarray], List]:
    """preprocess

    Args:
        video_path (str): input video path

    Returns:
        Tuple[Dict[str, np.ndarray], List]: feed and fetch
    """
    seq = Compose([
        VideoDecoder(),
        Sampler(8, 1, valid_mode=True),
        Scale(256),
        CenterCrop(224),
        Image2Array(),
        Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    results = {"filename": video_path}
    results = seq(results)
    tmp_inp = np.expand_dims(results["imgs"], axis=0)  # [b,t,c,h,w]
    tmp_inp = np.expand_dims(tmp_inp, axis=0)  # [1,b,t,c,h,w]
    feed = {"data_batch_0": tmp_inp}
    fetch = ["outputs"]
    return feed, fetch


def np_softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    """softmax function

    Args:
        x (np.ndarray): logits
        axis (int): axis

    Returns:
        np.ndarray: probs
    """
    x -= np.max(x, axis=axis, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    return x


def postprocess(fetch_map: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """postprocess

    Args:
        fetch_map (Dict[str, np.ndarray]): raw prediction

    Returns:
        Dict[str, Any]: postprocessed prediction
    """
    score_list = fetch_map["outputs"]  # [b,num_classes]
    fetch_dict = {"class_id": [], "prob": []}
    for score in score_list:
        score = np_softmax(score, axis=0)
        score = score.tolist()
        max_score = max(score)
        fetch_dict["class_id"].append(score.index(max_score))
        fetch_dict["prob"].append(max_score)

    fetch_dict["class_id"] = str(fetch_dict["class_id"])
    fetch_dict["prob"] = str(fetch_dict["prob"])
    return fetch_dict


def parse_args():
    # general params
    parser = argparse.ArgumentParser("PaddleVideo CPP Serving model script")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="serving client config file(serving_client_conf.prototxt) path")
    parser.add_argument("--url",
                        type=str,
                        default="127.0.0.1:9993",
                        help="url to access cpp serving")
    parser.add_argument("--logid", type=int, default="10000", help="log id")
    parser.add_argument("--input_file",
                        type=str,
                        default="../../data/example.avi",
                        help="input video file")
    return parser.parse_args()


if __name__ == "__main__":
    # parse args
    args = parse_args()
    url = args.url
    logid = args.logid
    input_file_path = args.input_file

    # initialize client object & connect
    client = Client()
    client.load_client_config(args.config)
    client.connect([url])

    # preprocess
    feed, fetch = preprocess(input_file_path)

    # send data & get prediction from server
    fetch_map = client.predict(feed=feed, fetch=fetch)

    # postprocess & output
    result = postprocess(fetch_map)
    print(result)
