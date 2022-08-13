# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import argparse
import numpy as np


from paddle import inference
from paddle.inference import Config, create_predictor
import dataset
from dataset.graphs import Graph

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser("efficientgcnv1 inference model script")
    parser.add_argument("--data_file", default='./data/ntu/tiny_dataset/tiny_infer_data.npy', type=str, help="input data path")
    parser.add_argument("--label_file", default='./data/ntu/tiny_dataset/tiny_infer_label.pkl', type=str, help="input label path")
    parser.add_argument("--model_file", default='./pretrain_models/xview.pdmodel', type=str)
    parser.add_argument("--params_file", default='./pretrain_models/xview.pdiparams', type=str)

    # params for predict
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("--use-gpu", type=str2bool, default=False)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--benchmark", type=str2bool, default=True)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument("--inputs", default='JVB', type=str)
    parser.add_argument('--data_shape', '-ds', type=int, nargs='+', default= [3, 6, 288, 25, 2], help='Using GPUs')
    parser.add_argument('--dataset', '-d', type=str, default= 'ntu', help='dataset')#
    return parser.parse_args()


def create_paddle_predictor(args):
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
        num_seg = 1
        num_views = 1
        max_batch_size = args.batch_size * num_views * num_seg
        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=max_batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return config, predictor


def parse_file_paths(args, data_path, label_path, use_mmap=True):
    try:
        with open(label_path) as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
                sample_name, label,seq_len = pickle.load(f, encoding='latin1')
            # sample_name, label = pickle.load(f, encoding='latin1')

    # load data
    if use_mmap:
        data = np.load(data_path, mmap_mode='r')
    else:
        data = np.load(data_path)
    _, C, T, V, M = args.data_shape

    data_new = []
    for i, d in enumerate(data):
        joint, velocity, bone = multi_input(args, d[:,:T,:,:])
        d_new = []
        if 'J' in args.inputs:
            d_new.append(joint)
        if 'V' in args.inputs:
            d_new.append(velocity)
        if 'B' in args.inputs:
            d_new.append(bone)
        d_new = np.stack(d_new, axis=0)
        data_new.append(d_new)
    data_new = np.array(data_new, dtype=np.float32)
    return data_new, sample_name, label
def multi_input(args, data):
    C, T, V, M = data.shape
    joint = np.zeros((C*2, T, V, M))
    velocity = np.zeros((C*2, T, V, M))
    bone = np.zeros((C*2, T, V, M))
    joint[:C,:,:,:] = data
    graph = Graph(args.dataset)
    for i in range(V):
        joint[C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
    for i in range(T-2):
        velocity[:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
        velocity[C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
    for i in range(len(graph.connect_joint)):
        bone[:C,:,i,:] = data[:,:,i,:] - data[:,:,graph.connect_joint[i],:]
    bone_length = 0
    for i in range(C):
        bone_length += bone[i,:,:,:] ** 2
    bone_length = np.sqrt(bone_length) + 0.0001
    for i in range(C):
        bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length)
    return joint, velocity, bone


def main():
    args = parse_args()
    model_name = 'EfficientGCNV1B0'
    print(f"Inference model({model_name})...")
    # InferenceHelper = build_inference_helper(cfg.INFERENCE)

    inference_config, predictor = create_paddle_predictor(args)
    # get data
    data, sample_name, label = parse_file_paths(args, data_path=args.data_file, label_path=args.label_file)
    # data = data[-100:]
    # sample_name = sample_name[-100:]
    # label = label[-100:]

    # Inferencing process
    batch_num = args.batch_size
    acc = []
    for st_idx in range(0, data.shape[0], batch_num):
        ed_idx = min(st_idx + batch_num, data.shape[0])

        # Pre process batched input
        batched_inputs = [data[st_idx:ed_idx]]
        batch_label = label[st_idx:ed_idx]
        batch_sample_name = sample_name[st_idx:ed_idx]

        # run inference
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(batched_inputs[i].shape)
            input_tensor.copy_from_cpu(batched_inputs[i].copy())

        # do the inference
        predictor.run()



        # get out data from output tensor
        results = []
        # get out data from output tensor
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        predict_label = np.argmax(results[0], 1)
        acc_batch = np.mean((predict_label == batch_label))
        acc.append(acc_batch)
        print('Batch action class Predict: ', predict_label,
              'Batch action class True: ', batch_label,
              'Batch Accuracy: ', acc_batch,
              'Batch sample Name: ', [name[-29:] for name in batch_sample_name])


    print('Infer Mean Accuracy: ', np.mean(np.array(acc)))


if __name__ == "__main__":
    main()
