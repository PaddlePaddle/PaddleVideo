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

import argparse

from utils import numpy_to_base64, parse_file_paths, video_to_numpy


def parse_args():
    # general params
    parser = argparse.ArgumentParser("PaddleVideo Web Serving model script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/PP-TSM.yaml',
                        help='serving config file path')
    parser.add_argument('-ptn',
                        '--port_number',
                        type=int,
                        default=9993,
                        help='rpc port number')
    parser.add_argument('-i',
                        '--input_file',
                        type=str,
                        help='input file path or directory path')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    client = PipelineClient()
    client.connect([f'127.0.0.1:{args.port_number}'])

    files_list = parse_file_paths(args.input_file)

    for file_path in files_list:
        # decoding video and get stacked frames as ndarray
        decoded_frames = video_to_numpy(file_path=file_path)

        # encode ndarray to base64 string for transportation.
        decoded_frames_base64 = numpy_to_base64(decoded_frames)

        # transport to server & get get results.
        ret = client.predict(feed_dict={
            "frames": decoded_frames_base64,
            "frames_shape": str(decoded_frames.shape)
        },
                             fetch=["label", "prob"])

        # print result
        print(ret)
