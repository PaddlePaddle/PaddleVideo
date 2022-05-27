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

import base64
import os
import os.path as osp

import cv2
import numpy as np


def numpy_to_base64(array: np.ndarray) -> str:
    """numpy_to_base64

    Args:
        array (np.ndarray): input ndarray.

    Returns:
        bytes object: encoded str.
    """
    return base64.b64encode(array).decode('utf8')


def video_to_numpy(file_path: str) -> np.ndarray:
    """decode video with cv2 and return stacked frames
       as numpy.

    Args:
        file_path (str): video file path.

    Returns:
        np.ndarray: [T,H,W,C] in uint8.
    """
    cap = cv2.VideoCapture(file_path)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_frames = []
    for i in range(videolen):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret is False:
            continue
        img = frame[:, :, ::-1]
        decoded_frames.append(img)
    decoded_frames = np.stack(decoded_frames, axis=0)
    return decoded_frames


def parse_file_paths(input_path: str) -> list:
    """get data pathes from input_path

    Args:
        input_path (str): input file path or directory which contains input file(s).

    Returns:
        list: path(es) of input file(s)
    """
    assert osp.exists(input_path), \
        f"{input_path} did not exists!"
    if osp.isfile(input_path):
        files = [
            input_path,
        ]
    else:
        files = os.listdir(input_path)
        files = [
            file for file in files
            if (file.endswith(".avi") or file.endswith(".mp4"))
        ]
        files = [osp.join(input_path, file) for file in files]
    return files
