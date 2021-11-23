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
import time

import numpy as np
from paddle import inference
from paddle.inference import Config, create_predictor

from utils import build_inference_helper
from paddlevideo.utils import get_config


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
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--params_file", type=str)

    # params for predict
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--enable_mkldnn", type=bool, default=False)
    parser.add_argument("--cpu_threads", type=int)
    # parser.add_argument("--hubserving", type=str2bool, default=False)  #TODO

    return parser.parse_args()


def create_paddle_predictor(args):
    config = Config(args.model_file, args.params_file)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if hasattr(args, "cpu_threads"):
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    #config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        # choose precision
        if args.precision == "fp16":
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=args.batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return config, predictor


def main():
    args = parse_args()
    cfg = get_config(args.config, show=False)

    model_name = cfg.model_name
    print(f"Inference model({model_name})...")
    InferenceHelper = build_inference_helper(cfg.INFERENCE)

    inference_config, predictor = create_paddle_predictor(args)

    # get input_tensor and output_tensor
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_tensor_list = []
    output_tensor_list = []
    for item in input_names:
        input_tensor_list.append(predictor.get_input_handle(item))
    for item in output_names:
        output_tensor_list.append(predictor.get_output_handle(item))

    if not args.enable_benchmark:
        # Pre process input
        inputs = InferenceHelper.preprocess(args.input_file)

        # Run inference
        for i in range(len(input_tensor_list)):
            input_tensor_list[i].copy_from_cpu(inputs[i])
        predictor.run()
        output = []
        for j in range(len(output_tensor_list)):
            output.append(output_tensor_list[j].copy_to_cpu())

        # Post process output
        InferenceHelper.postprocess(output)
    else:
        test_num = 500
        test_time = 0.0
        log_interval = 20
        num_warmup = 10

        # instantiate auto log
        import auto_log
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
            gpu_ids=0,
            time_keys=['preprocess_time', 'inference_time', 'postprocess_time'],
            warmup=num_warmup)

        for i in range(0, test_num + num_warmup):
            if (i + 1) % log_interval == 0 or (i + 1) == test_num + num_warmup:
                print(f"Benchmark process {i + 1}/{test_num + num_warmup}")
            input_list = []

            start_time = time.time()
            # auto log start
            if args.enable_benchmark:
                autolog.times.start()

            # Pre process input
            batched_inputs_list = []
            batch_count = 0
            while batch_count < args.batch_size:
                inputs = InferenceHelper.preprocess(args.input_file)
                batched_inputs_list.append(inputs)
                if 'tsm' in cfg.model_name.lower():
                    batch_count += (inputs[0].shape[1] * 1)  # centercrop
                elif 'tsn' in cfg.model_name.lower():
                    batch_count += (inputs[0].shape[1] * 10)  # tencrop
                elif 'timesformer' in cfg.model_name.lower():
                    batch_count += (inputs[0].shape[2] * 3)  # threecrop
                else:
                    batch_count += inputs[0].shape[0]

            batched_inputs = np.concatenate(batched_inputs_list, axis=0)
            input_list.extend(batched_inputs)

            # get pre process time cost
            if args.enable_benchmark:
                autolog.times.stamp()

            for j in range(len(input_tensor_list)):
                input_tensor_list[j].copy_from_cpu(input_list[j])

            predictor.run()

            output = []
            for j in range(len(output_tensor_list)):
                output.append(output_tensor_list[j].copy_to_cpu())

            # get inference process time cost
            if args.enable_benchmark:
                autolog.times.stamp()

            InferenceHelper.postprocess(output, False)

            # get post process time cost
            if args.enable_benchmark:
                autolog.times.end(stamp=True)

            if i >= 10:
                test_time += time.time() - start_time
            # time.sleep(0.01)  # sleep for T4 GPU

        # report benchmark log if enabled
        if args.enable_benchmark:
            autolog.report()


if __name__ == "__main__":
    main()
