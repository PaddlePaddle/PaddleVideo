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
import os
import sys
from os import path as osp

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../tools')))

from utils import build_inference_helper, get_config


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser("PaddleVideo Inference model script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument("-i", "--input_file", type=str, help="input file path")
    parser.add_argument("--onnx_file", type=str, help="onnx model file path")

    # params for onnx predict
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_gpu",
                        type=str2bool,
                        default=False,
                        help="set to False when using onnx")
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--enable_benchmark",
                        type=str2bool,
                        default=False,
                        help="set to False when using onnx")
    parser.add_argument("--cpu_threads", type=int, default=4)

    return parser.parse_args()


def create_onnx_predictor(args, cfg=None):
    import onnxruntime as ort
    onnx_file = args.onnx_file
    config = ort.SessionOptions()
    if args.use_gpu:
        raise ValueError(
            "onnx inference now only supports cpu! please set `use_gpu` to False."
        )
    else:
        config.intra_op_num_threads = args.cpu_threads
        if args.ir_optim:
            config.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    predictor = ort.InferenceSession(onnx_file, sess_options=config)
    return config, predictor


def parse_file_paths(input_path: str) -> list:
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


def main():
    """predict using onnx model
    """
    args = parse_args()
    cfg = get_config(args.config, show=False)

    model_name = cfg.model_name

    print(f"Inference model({model_name})...")
    InferenceHelper = build_inference_helper(cfg.INFERENCE)

    inference_config, predictor = create_onnx_predictor(args)

    # get input_tensor and output_tensor
    input_names = predictor.get_inputs()[0].name
    output_names = predictor.get_outputs()[0].name

    # get the absolute file path(s) to be processed
    files = parse_file_paths(args.input_file)
    if args.enable_benchmark:
        test_video_num = 12
        num_warmup = 3
        # instantiate auto log
        try:
            import auto_log
        except ImportError as e:
            print(f"{e}, [git+https://github.com/LDOUBLEV/AutoLog] "
                  f"package and it's dependencies is required for "
                  f"python-inference when enable_benchmark=True.")
        pid = os.getpid()
        autolog = auto_log.AutoLogger(
            model_name=cfg.model_name,
            model_precision=args.precision,
            batch_size=args.batch_size,
            data_shape="dynamic",
            save_path="./output/auto_log.lpg",
            inference_config=inference_config,
            pids=pid,
            process_name=None,
            gpu_ids=None,
            time_keys=['preprocess_time', 'inference_time', 'postprocess_time'],
            warmup=num_warmup)
        files = [args.input_file for _ in range(test_video_num + num_warmup)]

    # Inferencing process
    batch_num = args.batch_size
    for st_idx in range(0, len(files), batch_num):
        ed_idx = min(st_idx + batch_num, len(files))

        # auto log start
        if args.enable_benchmark:
            autolog.times.start()

        # Pre process batched input
        batched_inputs = InferenceHelper.preprocess_batch(files[st_idx:ed_idx])

        # get pre process time cost
        if args.enable_benchmark:
            autolog.times.stamp()

        # run inference
        batched_outputs = predictor.run(
            output_names=[output_names],
            input_feed={input_names: batched_inputs[0]})

        # get inference process time cost
        if args.enable_benchmark:
            autolog.times.stamp()

        InferenceHelper.postprocess(batched_outputs, not args.enable_benchmark)

        # get post process time cost
        if args.enable_benchmark:
            autolog.times.end(stamp=True)

        # time.sleep(0.01)  # sleep for T4 GPU

    # report benchmark log if enabled
    if args.enable_benchmark:
        autolog.report()


if __name__ == "__main__":
    main()
