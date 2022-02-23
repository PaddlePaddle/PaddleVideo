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
try:
    from paddle_serving_server_gpu.pipeline import PipelineClient
except ImportError:
    from paddle_serving_server.pipeline import PipelineClient

import base64

import cv2
import numpy as np

client = PipelineClient()
client.connect(['127.0.0.1:9993'])


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


if __name__ == "__main__":
    video_path = '../../data/example.avi'

    # decoding video and get stacked frames as ndarray
    decoded_frames = video_to_numpy(file_path=video_path)

    # encode ndarray to base64 string for transportation.
    decoded_frames_base64 = numpy_to_base64(decoded_frames)

    # transport to server & get get results.
    for i in range(1):
        ret = client.predict(feed_dict={
            "frames": decoded_frames_base64,
            "frames_shape": str(decoded_frames.shape)
        },
                             fetch=["label", "prob"])
        print(ret)
