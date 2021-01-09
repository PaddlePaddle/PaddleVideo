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

import os
import sys

__dir__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__dir__, ''))

import cv2
import numpy as np
import tarfile
import requests
from tqdm import tqdm
from tools import utils
import shutil

from paddle.inference import Config
from paddle.inference import create_predictor

__all__ = ['PaddleVideo']
BASE_DIR = os.path.expanduser("~/.paddlevideo_inference/")
BASE_INFERENCE_MODEL_DIR = os.path.join(BASE_DIR, 'inference_model')
BASE_IMAGES_DIR = os.path.join(BASE_DIR, 'images')

model_names = {'TSM','TSN','PPTSM','SlowFast'}


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


def download_with_progressbar(url, save_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes == 0 or progress_bar.n != total_size_in_bytes:
        raise Exception("Something went wrong while downloading models")


def maybe_download(model_storage_directory, url):
    # using custom model
    tar_file_name_list = [
        'inference.pdiparams', 'inference.pdiparams.info', 'inference.pdmodel' #pdiparams,和pdmodel直接下载
    ]
    if not os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdiparams')
    ) or not os.path.exists(
        os.path.join(model_storage_directory, 'inference.pdmodel')):
        tmp_path = os.path.join(model_storage_directory, url.split('/')[-1])
        print('download {} to {}'.format(url, tmp_path))
        os.makedirs(model_storage_directory, exist_ok=True)
        download_with_progressbar(url, tmp_path) #download

        #save to directory
        with tarfile.open(tmp_path, 'r') as tarObj:
            for member in tarObj.getmembers():
                filename = None
                for tar_file_name in tar_file_name_list:
                    if tar_file_name in member.name:
                        filename = tar_file_name
                if filename is None:
                    continue
                file = tarObj.extractfile(member)
                with open(
                        os.path.join(model_storage_directory, filename),
                        'wb') as f:
                    f.write(file.read())
        os.remove(tmp_path)


def save_prelabel_results(class_id, input_filepath, output_idr):
    output_dir = os.path.join(output_idr, str(class_id))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    shutil.copy(input_filepath, output_dir)


def load_label_name_dict(path):
    result = {}
    if not os.path.exists(path):
        print(
            'Warning: If want to use your own label_dict, please input legal path!\nOtherwise label_names will be empty!'
        )
    else:
        for line in open(path, 'r'):
            partition = line.split('\n')[0].partition(' ')
            try:
                result[int(partition[0])] = str(partition[-1])
            except:
                result = {}
                break
    return result

# def parse_args():
#     def str2bool(v):
#         return v.lower() in ("true", "t", "1")
#
#     import argparse
#     # general params
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-v", "--video_file", type=str, help="video file path")
#     parser.add_argument("--use_gpu", type=str2bool, default=True)
#
#     # params for decode and sample
#     parser.add_argument("--num_seg", type=int, default=8)
#     parser.add_argument("--seg_len", type=int, default=1)
#
#     # params for preprocess
#     parser.add_argument("--short_size", type=int, default=256)
#     parser.add_argument("--target_size", type=int, default=224)
#     parser.add_argument("--normalize", type=str2bool, default=True)
#
#     # params for predict
#     parser.add_argument("--model_file", type=str)
#     parser.add_argument("--params_file", type=str)
#     parser.add_argument("-b", "--batch_size", type=int, default=1)
#     parser.add_argument("--use_fp16", type=str2bool, default=False)
#     parser.add_argument("--ir_optim", type=str2bool, default=True)
#     parser.add_argument("--use_tensorrt", type=str2bool, default=False)
#     parser.add_argument("--gpu_mem", type=int, default=8000)
#     parser.add_argument("--enable_benchmark", type=str2bool, default=False)
#     parser.add_argument("--top_k", type=int, default=1)
#     parser.add_argument("--enable_mkldnn", type=bool, default=False)
#     parser.add_argument("--hubserving", type=str2bool, default=False)
#
#     # params for infer
#
#     parser.add_argument("--model", type=str)
#     """
#     parser.add_argument("--pretrained_model", type=str)
#     parser.add_argument("--class_num", type=int, default=1000)
#     parser.add_argument(
#         "--load_static_weights",
#         type=str2bool,
#         default=False,
#         help='Whether to load the pretrained weights saved in static mode')
#
#     # parameters for pre-label the images
#     parser.add_argument(
#         "--pre_label_image",
#         type=str2bool,
#         default=False,
#         help="Whether to pre-label the images using the loaded weights")
#     parser.add_argument("--pre_label_out_idr", type=str, default=None)
#     """
#
#     return parser.parse_args()

class PaddleVideo(object):
    print('Inference models that Paddle provides are listed as follows:\n\n{}'.
          format(model_names), '\n')

    def __init__(self, **kwargs):
        process_params = utils.parse_args()
        process_params.__dict__.update(**kwargs)
        print(process_params)

        if not os.path.exists(process_params.model_file):
            if process_params.model is None:
                raise Exception(
                    'Please input model name that you want to use!')
            if process_params.model in model_names:
                url = 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/{}_infer.tar'.format(
                    process_params.model)
                print('----221------',os.path.join(BASE_INFERENCE_MODEL_DIR,
                                                     process_params.model))
                if not os.path.exists(
                        os.path.join(BASE_INFERENCE_MODEL_DIR,
                                     process_params.model)):
                    os.makedirs(
                        os.path.join(BASE_INFERENCE_MODEL_DIR,
                                     process_params.model))
                #create pretrained model download_path
                download_path = os.path.join(BASE_INFERENCE_MODEL_DIR,
                                             process_params.model)
                maybe_download(model_storage_directory=download_path, url=url)
                process_params.model_file = os.path.join(download_path,
                                                         'inference.pdmodel')
                process_params.params_file = os.path.join(
                    download_path, 'inference.pdiparams')
                process_params.label_name_path = os.path.join(
                    __dir__, 'ppcls/utils/imagenet1k_label_list.txt') #需要提供一个label_list
            else:
                raise Exception(
                    'If you want to use your own model, Please input model_file as model path!'
                )
        else:
            print('Using user-specified model and params!')
        print("process params are as follows: \n{}".format(process_params))#一个字典
        self.label_name_dict = load_label_name_dict(
            process_params.label_name_path)

        self.args = process_params
        self.predictor = create_paddle_predictor(process_params)

    def predict(self):

        assert self.args.batch_size == 1
        assert self.args.use_fp16 is False

        input_names = self.predictor.get_input_names()
        input_tensor = self.predictor.get_input_handle(input_names[0])

        output_names = self.predictor.get_output_names()
        output_tensor = self.predictor.get_output_handle(output_names[0])

        if not self.args.enable_benchmark:
            # for PaddleHubServing
            if self.args.hubserving:
                img = self.args.image_file
            # for predict only
            else:
                # img = cv2.imread(args.image_file)[:, :, ::-1]
                img = utils.decode(self.args.video_file, self.args)
            assert img is not None, "Error in loading video: {}".format(
                self.args.video_file)
            inputs = utils.preprocess(img, self.args)
            inputs = np.expand_dims(
                inputs, axis=0).repeat(
                self.args.batch_size, axis=0).copy()

            input_tensor.copy_from_cpu(inputs)

            self.predictor.run()

            output = output_tensor.copy_to_cpu()
            return utils.postprocess(output, self.args)

if __name__ == '__main__':
    cls = PaddleVideo(model='TSN')
