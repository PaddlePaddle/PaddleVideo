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
import numpy as np
import time

from utils import build_inference_helper
from paddle.inference import Config
from paddle.inference import create_predictor
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
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--enable_mkldnn", type=bool, default=False)
    # parser.add_argument("--hubserving", type=str2bool, default=False)  #TODO

    return parser.parse_args()


def create_paddle_predictor(args):
    config = Config(args.model_file, args.params_file)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()

    #config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        config.enable_tensorrt_engine(
            precision_mode=Config.Precision.Half
            if args.use_fp16 else Config.Precision.Float32,
            max_batch_size=args.batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return predictor


def main():
    args = parse_args()
    cfg = get_config(args.config, show=False)
    model_name = cfg.model_name
    print(f"Inference model({model_name})...")
    InferenceHelper = build_inference_helper(cfg.INFERENCE)

    if args.enable_benchmark:
        assert args.use_gpu is True

    # HALF precission predict only work when using tensorrt
    if args.use_fp16 is True:
        assert args.use_tensorrt is True

    predictor = create_paddle_predictor(args)

    # get input_tensor and output_tensor
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_tensor_list = []
    output_tensor_list = []
    for item in input_names:
        input_tensor_list.append(predictor.get_input_handle(item))
    for item in output_names:
        output_tensor_list.append(predictor.get_output_handle(item))

    test_num = 500
    test_time = 0.0
    if not args.enable_benchmark:
        # Prepare input
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
    else:  # benchmark only for ppTSM
        for i in range(0, test_num + 10):
            inputs = []
            inputs.append(
                np.random.rand(args.batch_size, 8, 3, 224,
                               224).astype(np.float32))
            start_time = time.time()
            for j in range(len(input_tensor_list)):
                input_tensor_list[j].copy_from_cpu(inputs[j])

            predictor.run()

            output = []
            for j in range(len(output_tensor_list)):
                output.append(output_tensor_list[j].copy_to_cpu())

            if i >= 10:
                test_time += time.time() - start_time
            #time.sleep(0.01)  # sleep for T4 GPU

        fp_message = "FP16" if args.use_fp16 else "FP32"
        trt_msg = "using tensorrt" if args.use_tensorrt else "not using tensorrt"
        print("{0}\t{1}\t{2}\tbatch size: {3}\ttime(ms): {4}".format(
            model_name, trt_msg, fp_message, args.batch_size,
            1000 * test_time / test_num))


if __name__ == "__main__":
    main()
