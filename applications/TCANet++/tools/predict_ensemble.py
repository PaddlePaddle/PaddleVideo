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
from os import path as osp

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

    parser.add_argument('-c1',
                        '--config1',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')

    parser.add_argument("-i", "--input_file", type=str, help="input file path")

    parser.add_argument("--model_file", type=str)
    parser.add_argument("--params_file", type=str)

    parser.add_argument("--model_file_1", type=str)
    parser.add_argument("--params_file_1", type=str)

    parser.add_argument("--model_file_2", type=str)
    parser.add_argument("--params_file_2", type=str)

    parser.add_argument("--model_file_3", type=str)
    parser.add_argument("--params_file_3", type=str)

    parser.add_argument("--model_file_4", type=str)
    parser.add_argument("--params_file_4", type=str)

    parser.add_argument("--model_file_5", type=str)
    parser.add_argument("--params_file_5", type=str)

    parser.add_argument("--model_file_6", type=str)
    parser.add_argument("--params_file_6", type=str)

    # params for predict
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=None)
    # parser.add_argument("--hubserving", type=str2bool, default=False)  #TODO

    return parser.parse_args()


def create_paddle_predictor(args, cfg):
    config = Config(args.model_file, args.params_file)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        # choose precision
        if args.precision == "fp16":
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        # calculate real max batch size during inference when tenrotRT enabled
        max_batch_size = args.batch_size
        if 'num_seg' in cfg.INFERENCE:
            num_seg = cfg.INFERENCE.num_seg
            num_views = 1
            if 'tsm' in cfg.model_name.lower():
                num_views = 1  # CenterCrop
            elif 'tsn' in cfg.model_name.lower():
                num_views = 10  # TenCrop
            elif 'timesformer' in cfg.model_name.lower():
                num_views = 3  # UniformCrop
            max_batch_size = args.batch_size * num_views * num_seg
        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=max_batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)

    # for ST-GCN tensorRT case usage
    # config.delete_pass("shuffle_channel_detect_pass")

    predictor = create_predictor(config)

    return config, predictor


def create_paddle_predictor_1(args, cfg):
    config = Config(args.model_file_1, args.params_file_1)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        # choose precision
        if args.precision == "fp16":
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        # calculate real max batch size during inference when tenrotRT enabled
        max_batch_size = args.batch_size
        if 'num_seg' in cfg.INFERENCE:
            num_seg = cfg.INFERENCE.num_seg
            num_views = 1
            if 'tsm' in cfg.model_name.lower():
                num_views = 1  # CenterCrop
            elif 'tsn' in cfg.model_name.lower():
                num_views = 10  # TenCrop
            elif 'timesformer' in cfg.model_name.lower():
                num_views = 3  # UniformCrop
            max_batch_size = args.batch_size * num_views * num_seg
        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=max_batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)

    # for ST-GCN tensorRT case usage
    # config.delete_pass("shuffle_channel_detect_pass")

    predictor = create_predictor(config)

    return config, predictor


def create_paddle_predictor_2(args, cfg):
    config = Config(args.model_file_2, args.params_file_2)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        # choose precision
        if args.precision == "fp16":
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        # calculate real max batch size during inference when tenrotRT enabled
        max_batch_size = args.batch_size
        if 'num_seg' in cfg.INFERENCE:
            num_seg = cfg.INFERENCE.num_seg
            num_views = 1
            if 'tsm' in cfg.model_name.lower():
                num_views = 1  # CenterCrop
            elif 'tsn' in cfg.model_name.lower():
                num_views = 10  # TenCrop
            elif 'timesformer' in cfg.model_name.lower():
                num_views = 3  # UniformCrop
            max_batch_size = args.batch_size * num_views * num_seg
        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=max_batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)

    # for ST-GCN tensorRT case usage
    # config.delete_pass("shuffle_channel_detect_pass")

    predictor = create_predictor(config)

    return config, predictor


def create_paddle_predictor_3(args, cfg):
    config = Config(args.model_file_3, args.params_file_3)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        # choose precision
        if args.precision == "fp16":
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        # calculate real max batch size during inference when tenrotRT enabled
        max_batch_size = args.batch_size
        if 'num_seg' in cfg.INFERENCE:
            num_seg = cfg.INFERENCE.num_seg
            num_views = 1
            if 'tsm' in cfg.model_name.lower():
                num_views = 1  # CenterCrop
            elif 'tsn' in cfg.model_name.lower():
                num_views = 10  # TenCrop
            elif 'timesformer' in cfg.model_name.lower():
                num_views = 3  # UniformCrop
            max_batch_size = args.batch_size * num_views * num_seg
        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=max_batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)

    # for ST-GCN tensorRT case usage
    # config.delete_pass("shuffle_channel_detect_pass")

    predictor = create_predictor(config)

    return config, predictor


def create_paddle_predictor_4(args, cfg):
    config = Config(args.model_file_4, args.params_file_4)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        # choose precision
        if args.precision == "fp16":
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        # calculate real max batch size during inference when tenrotRT enabled
        max_batch_size = args.batch_size
        if 'num_seg' in cfg.INFERENCE:
            num_seg = cfg.INFERENCE.num_seg
            num_views = 1
            if 'tsm' in cfg.model_name.lower():
                num_views = 1  # CenterCrop
            elif 'tsn' in cfg.model_name.lower():
                num_views = 10  # TenCrop
            elif 'timesformer' in cfg.model_name.lower():
                num_views = 3  # UniformCrop
            max_batch_size = args.batch_size * num_views * num_seg
        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=max_batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)

    # for ST-GCN tensorRT case usage
    # config.delete_pass("shuffle_channel_detect_pass")

    predictor = create_predictor(config)

    return config, predictor


def create_paddle_predictor_5(args, cfg):
    config = Config(args.model_file_5, args.params_file_5)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        # choose precision
        if args.precision == "fp16":
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        # calculate real max batch size during inference when tenrotRT enabled
        max_batch_size = args.batch_size
        if 'num_seg' in cfg.INFERENCE:
            num_seg = cfg.INFERENCE.num_seg
            num_views = 1
            if 'tsm' in cfg.model_name.lower():
                num_views = 1  # CenterCrop
            elif 'tsn' in cfg.model_name.lower():
                num_views = 10  # TenCrop
            elif 'timesformer' in cfg.model_name.lower():
                num_views = 3  # UniformCrop
            max_batch_size = args.batch_size * num_views * num_seg
        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=max_batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)

    # for ST-GCN tensorRT case usage
    # config.delete_pass("shuffle_channel_detect_pass")

    predictor = create_predictor(config)

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
            if (file.endswith(".avi") or file.endswith(".mp4") or file.endswith(".npy"))
        ]
        files = [osp.join(input_path, file) for file in files]
    return files


def main():
    args = parse_args()
    cfg = get_config(args.config, show=False)
    cfg1 = get_config(args.config1, show=False)

    model_name = cfg.model_name
    model_name1 = cfg1.model_name

    print(f"Inference model({model_name})...")
    print(f"Inference model({model_name1})...")

    InferenceHelper = build_inference_helper(cfg.INFERENCE)
    InferenceHelper1 = build_inference_helper(cfg1.INFERENCE)

    inference_config, predictor = create_paddle_predictor(args, cfg)
    inference_config_1, predictor_1 = create_paddle_predictor_1(args, cfg)
    inference_config_2, predictor_2 = create_paddle_predictor_2(args, cfg)
    inference_config_3, predictor_3 = create_paddle_predictor_3(args, cfg)
    inference_config_4, predictor_4 = create_paddle_predictor_4(args, cfg1)
    inference_config_5, predictor_5 = create_paddle_predictor_5(args, cfg1)

    # get input_tensor and output_tensor
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_tensor_list = []
    output_tensor_list = []
    for item in input_names:
        input_tensor_list.append(predictor.get_input_handle(item))
    for item in output_names:
        output_tensor_list.append(predictor.get_output_handle(item))

    # get input_tensor and output_tensor
    input_names_1 = predictor_1.get_input_names()
    output_names_1 = predictor_1.get_output_names()
    input_tensor_list_1 = []
    output_tensor_list_1 = []
    for item in input_names_1:
        input_tensor_list_1.append(predictor_1.get_input_handle(item))
    for item in output_names_1:
        output_tensor_list_1.append(predictor_1.get_output_handle(item))

    # get input_tensor and output_tensor
    input_names_2 = predictor_2.get_input_names()
    output_names_2 = predictor_2.get_output_names()
    input_tensor_list_2 = []
    output_tensor_list_2 = []
    for item in input_names_2:
        input_tensor_list_2.append(predictor_2.get_input_handle(item))
    for item in output_names_2:
        output_tensor_list_2.append(predictor_2.get_output_handle(item))

    # get input_tensor and output_tensor
    input_names_3 = predictor_3.get_input_names()
    output_names_3 = predictor_3.get_output_names()
    input_tensor_list_3 = []
    output_tensor_list_3 = []
    for item in input_names_3:
        input_tensor_list_3.append(predictor_3.get_input_handle(item))
    for item in output_names_3:
        output_tensor_list_3.append(predictor_3.get_output_handle(item))

    files = parse_file_paths(args.input_file)

    # get input_tensor and output_tensor
    input_names_4 = predictor_4.get_input_names()
    output_names_4 = predictor_4.get_output_names()
    input_tensor_list_4 = []
    output_tensor_list_4 = []
    for item in input_names_4:
        input_tensor_list_4.append(predictor_4.get_input_handle(item))
    for item in output_names_4:
        output_tensor_list_4.append(predictor_4.get_output_handle(item))

    # get input_tensor and output_tensor
    input_names_5 = predictor_5.get_input_names()
    output_names_5 = predictor_5.get_output_names()
    input_tensor_list_5 = []
    output_tensor_list_5 = []
    for item in input_names_5:
        input_tensor_list_5.append(predictor_5.get_input_handle(item))
    for item in output_names_5:
        output_tensor_list_5.append(predictor_5.get_output_handle(item))

    files = parse_file_paths(args.input_file)

    # Inferencing process
    batch_num = args.batch_size
    for st_idx in range(0, len(files), batch_num):
        ed_idx = min(st_idx + batch_num, len(files))

        # Pre process batched input
        batched_inputs = InferenceHelper.preprocess_batch(files[st_idx:ed_idx])
        batched_inputs1 = InferenceHelper1.preprocess_batch(files[st_idx:ed_idx])

        # run inference 1
        for i in range(len(input_tensor_list)):
            input_tensor_list[i].copy_from_cpu(batched_inputs[i])
        predictor.run()
        batched_outputs1 = []
        for j in range(len(output_tensor_list)):
            batched_outputs1.append(output_tensor_list[j].copy_to_cpu())

        # run inference 2
        for i in range(len(input_tensor_list_1)):
            input_tensor_list_1[i].copy_from_cpu(batched_inputs[i])
        predictor_1.run()
        batched_outputs2 = []
        for j in range(len(output_tensor_list_1)):
            batched_outputs2.append(output_tensor_list_1[j].copy_to_cpu())

        # run inference 3
        for i in range(len(input_tensor_list_2)):
            input_tensor_list_2[i].copy_from_cpu(batched_inputs[i])
        predictor_2.run()
        batched_outputs3 = []
        for j in range(len(output_tensor_list_2)):
            batched_outputs3.append(output_tensor_list_2[j].copy_to_cpu())

        # run inference 4
        for i in range(len(input_tensor_list_3)):
            input_tensor_list_3[i].copy_from_cpu(batched_inputs[i])
        predictor_3.run()
        batched_outputs4 = []
        for j in range(len(output_tensor_list_3)):
            batched_outputs4.append(output_tensor_list_3[j].copy_to_cpu())

        # run inference 5
        for i in range(len(input_tensor_list_4)):
            input_tensor_list_4[i].copy_from_cpu(batched_inputs1[i])
        predictor_4.run()
        batched_outputs5 = []
        for j in range(len(output_tensor_list_4)):
            batched_outputs5.append(output_tensor_list_4[j].copy_to_cpu())

        # run inference 6
        for i in range(len(input_tensor_list_5)):
            input_tensor_list_5[i].copy_from_cpu(batched_inputs1[i])
        predictor_5.run()
        batched_outputs6 = []
        for j in range(len(output_tensor_list_5)):
            batched_outputs6.append(output_tensor_list_5[j].copy_to_cpu())

        InferenceHelper.postprocess(batched_outputs1,batched_outputs2,batched_outputs3,batched_outputs4,batched_outputs5,batched_outputs6,not args.enable_benchmark)


if __name__ == "__main__":
    main()