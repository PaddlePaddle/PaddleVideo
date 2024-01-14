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
import base64
import os
import sys
from typing import Callable, Dict, List

import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from paddle_serving_app.reader import Sequential
from paddlevideo.loader.pipelines import (CenterCrop, Image2Array,
                                          Normalization, Sampler, Scale,
                                          TenCrop)

try:
    from paddle_serving_server_gpu.web_service import Op, WebService
except ImportError:
    from paddle_serving_server.web_service import Op, WebService

VALID_MODELS = ["PPTSM", "PPTSN"]


def get_preprocess_seq(model_name: str) -> List[Callable]:
    """get preprocess sequence by model name

    Args:
        model_name (str): model name for web serving, such as 'PPTSM', 'PPTSN'

    Returns:
        List[Callable]: preprocess operators in list.
    """
    if model_name == 'PPTSM':
        preprocess_seq = [
            Sampler(8, 1, valid_mode=True),
            Scale(256),
            CenterCrop(224),
            Image2Array(),
            Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    elif model_name == 'PPTSN':
        preprocess_seq = [
            Sampler(25, 1, valid_mode=True, select_left=True),
            Scale(256, fixed_ratio=True, do_round=True, backend='cv2'),
            TenCrop(224),
            Image2Array(),
            Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    else:
        raise ValueError(
            f"model_name must in {VALID_MODELS}, but got {model_name}")
    return preprocess_seq


def np_softmax(x: np.ndarray, axis=0) -> np.ndarray:
    """softmax function

    Args:
        x (np.ndarray): logits.

    Returns:
        np.ndarray: probs.
    """
    x -= np.max(x, axis=axis, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    return x


class VideoOp(Op):
    def init_op(self):
        """init_op
        """
        self.seq = Sequential(get_preprocess_seq(args.name))

        self.label_dict = {}
        with open("../../data/k400/Kinetics-400_label_list.txt", "r") as fin:
            for line in fin:
                label_ind, label_name = line.strip().split(' ')
                label_ind = int(label_ind)
                self.label_dict[label_ind] = label_name.strip()

    def preprocess(self, input_dicts: Dict, data_id: int, log_id: int):
        """preprocess

        Args:
            input_dicts (Dict): input_dicts.
            data_id (int): data_id.
            log_id (int): log_id.

        Returns:
            output_data: data for process stage.
            is_skip_process: skip process stage or not, False default
            prod_errcode: None default, otherwise, product errores occured.
                          It is handled in the same way as exception.
            prod_errinfo: "" default.
        """
        (_, input_dict), = input_dicts.items()
        for key in input_dict.keys():
            if key == "frames":
                frame_data = base64.b64decode(input_dict[key].encode('utf8'))
                frame_data = np.fromstring(frame_data, np.uint8)
            elif key == 'frames_shape':
                shape_data = eval(input_dict[key])
            else:
                raise ValueError(f"unexpected key received: {key}")
        frame_data = frame_data.reshape(shape_data)
        frame_len = frame_data.shape[0]
        frame_data = np.split(frame_data, frame_len, axis=0)
        frame_data = [frame.squeeze(0) for frame in frame_data]
        results = {
            'frames': frame_data,
            'frames_len': frame_len,
            'format': 'video',
            'backend': 'cv2'
        }
        results = self.seq(results)
        tmp_inp = np.expand_dims(results['imgs'], axis=0)  # [b,t,c,h,w]

        # The input for the network is input_data[0], so need to add 1 dimension at the beginning
        tmp_inp = np.expand_dims(tmp_inp, axis=0).copy()  # [1,b,t,c,h,w]
        return {"data_batch_0": tmp_inp}, False, None, ""

    def postprocess(self, input_dicts: Dict, fetch_dict: Dict, data_id: int,
                    log_id: int):
        """postprocess

        Args:
            input_dicts (Dict): data returned in preprocess stage, dict(for single predict) or list(for batch predict).
            fetch_dict (Dict): data returned in process stage, dict(for single predict) or list(for batch predict).
            data_id (int): inner unique id, increase auto.
            log_id (int): logid, 0 default.

        Returns:
            fetch_dict: fetch result must be dict type.
            prod_errcode: None default, otherwise, product errores occured.
                          It is handled in the same way as exception.
            prod_errinfo: "" default.
        """
        score_list = fetch_dict["outputs"]
        result = {"label": [], "prob": []}
        for score in score_list:
            score = np_softmax(score)
            score = score.tolist()
            max_score = max(score)
            max_index = score.index(max_score)
            result["label"].append(self.label_dict[max_index])
            result["prob"].append(max_score)
        result["label"] = str(result["label"])
        result["prob"] = str(result["prob"])
        return result, None, ""


class VideoService(WebService):
    def get_pipeline_response(self, read_op):
        """get_pipeline_response

        Args:
            read_op ([type]): [description]

        Returns:
            [type]: [description]
        """
        video_op = VideoOp(name="video", input_ops=[read_op])
        return video_op


def parse_args():
    # general params
    parser = argparse.ArgumentParser("PaddleVideo Web Serving model script")
    parser.add_argument(
        '-n',
        '--name',
        type=str,
        default='PPTSM',
        help='model name used in web serving, such as PPTSM, PPTSN...')

    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/PP-TSM.yaml',
                        help='serving config file path')

    return parser.parse_args()


if __name__ == '__main__':
    # get args such as serving config yaml path.
    args = parse_args()

    # start serving
    uci_service = VideoService(name="video")
    uci_service.prepare_pipeline_config(yaml_file=args.config)
    uci_service.run_service()
