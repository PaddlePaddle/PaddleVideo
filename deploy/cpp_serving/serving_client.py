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
from typing import Any, Dict

import numpy as np
from paddle_serving_client import Client

from preprocess_ops import get_preprocess_func, np_softmax


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
    parser.add_argument("-n",
                        "--name",
                        type=str,
                        default="PPTSM",
                        help="model's name, such as PPTSM, PPTSN...")
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
    model_name = args.name

    # get preprocess by model name
    preprocess = get_preprocess_func(model_name)

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
