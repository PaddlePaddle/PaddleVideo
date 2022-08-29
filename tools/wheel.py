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

import numpy as np
import tarfile
import requests
from tqdm import tqdm
import shutil

from paddle import inference
from paddle.inference import Config, create_predictor

from tools.utils import ppTSM_Inference_helper

__all__ = ['PaddleVideo']

# path of download model and data
BASE_DIR = os.path.expanduser("~/.paddlevideo_inference/")
BASE_INFERENCE_MODEL_DIR = os.path.join(BASE_DIR, 'inference_model')
BASE_VIDEOS_DIR = os.path.join(BASE_DIR, 'videos')

# support Models
MODELS = {
    'ppTSM':
    'https://videotag.bj.bcebos.com/PaddleVideo/InferenceModel/ppTSM_infer.tar',
    'ppTSM_v2':
    'https://videotag.bj.bcebos.com/PaddleVideo/InferenceModel/ppTSM_v2_infer.tar'
}

MODEL_NAMES = list(MODELS.keys())


def parse_args(mMain=True, add_help=True):
    """
    Args:
        mMain: bool. True for command args, False for python interface
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    if mMain == True:

        # general params
        parser = argparse.ArgumentParser(add_help=add_help)
        parser.add_argument("--model_name", type=str, default='')
        parser.add_argument("-v", "--video_file", type=str, default='')
        parser.add_argument("--use_gpu", type=str2bool, default=True)

        # params for decode and sample
        parser.add_argument("--num_seg", type=int, default=16)

        # params for preprocess
        parser.add_argument("--short_size", type=int, default=256)
        parser.add_argument("--target_size", type=int, default=224)

        # params for predict
        parser.add_argument("--model_file", type=str, default='')
        parser.add_argument("--params_file", type=str)
        parser.add_argument("-b", "--batch_size", type=int, default=1)
        parser.add_argument("--use_fp16", type=str2bool, default=False)
        parser.add_argument("--ir_optim", type=str2bool, default=True)
        parser.add_argument("--use_tensorrt", type=str2bool, default=False)
        parser.add_argument("--gpu_mem", type=int, default=8000)
        parser.add_argument("--top_k", type=int, default=1)
        parser.add_argument("--enable_mkldnn", type=bool, default=False)
        parser.add_argument("--label_name_path", type=str, default='')

        return parser.parse_args()

    else:
        return argparse.Namespace(model_name='',
                                  video_file='',
                                  use_gpu=True,
                                  num_seg=16,
                                  short_size=256,
                                  target_size=224,
                                  model_file='',
                                  params_file='',
                                  batch_size=1,
                                  use_fp16=False,
                                  ir_optim=True,
                                  use_tensorrt=False,
                                  gpu_mem=8000,
                                  top_k=1,
                                  enable_mkldnn=False,
                                  label_name_path='')


def parse_file_paths(input_path: str) -> list:
    if os.path.isfile(input_path):
        files = [
            input_path,
        ]
    else:
        files = os.listdir(input_path)
        files = [
            file for file in files
            if (file.endswith(".avi") or file.endswith(".mp4"))
        ]
        files = [os.path.join(input_path, file) for file in files]
    return files


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


def download_inference_model(model_storage_directory, url):
    # using custom model
    tar_file_name_list = [
        'inference.pdiparams', 'inference.pdiparams.info', 'inference.pdmodel'
    ]
    if not os.path.exists(
            os.path.join(model_storage_directory,
                         'inference.pdiparams')) or not os.path.exists(
                             os.path.join(model_storage_directory,
                                          'inference.pdmodel')):
        tmp_path = os.path.join(model_storage_directory, url.split('/')[-1])
        print('download {} to {}'.format(url, tmp_path))
        os.makedirs(model_storage_directory, exist_ok=True)
        download_with_progressbar(url, tmp_path)  #download

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
                with open(os.path.join(model_storage_directory, filename),
                          'wb') as f:
                    f.write(file.read())
        os.remove(tmp_path)


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

    config.disable_glog_info()
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


class PaddleVideo(object):
    def __init__(self, **kwargs):
        print(
            '\nInference models that Paddle provides are listed as follows:\n{}'
            .format(MODEL_NAMES), '\n')
        process_params = parse_args(mMain=False, add_help=False)
        process_params.__dict__.update(**kwargs)

        if not os.path.exists(process_params.model_file):
            if process_params.model_name is None:
                raise Exception('Please input model name that you want to use!')
            if process_params.model_name in MODEL_NAMES:
                url = MODELS[process_params.model_name]
                download_path = os.path.join(BASE_INFERENCE_MODEL_DIR,
                                             process_params.model_name)
                if not os.path.exists(download_path):
                    os.makedirs(download_path)

                #create pretrained model download_path
                download_inference_model(model_storage_directory=download_path,
                                         url=url)

                process_params.model_file = os.path.join(
                    download_path, 'inference.pdmodel')
                process_params.params_file = os.path.join(
                    download_path, 'inference.pdiparams')
                process_params.label_name_path = os.path.join(
                    __dir__, '../data/k400/Kinetics-400_label_list.txt')
            else:
                raise Exception(
                    'If you want to use your own model, Please input model_file as model path!'
                )
        else:
            print('Using user-specified model and params!')
        print("process params are as follows: \n{}".format(process_params))
        self.label_name_dict = load_label_name_dict(
            process_params.label_name_path)

        self.args = process_params
        self.predictor = create_paddle_predictor(process_params)

    def predict(self, video):
        """
        predict label of video with paddlevideo
        Args:
            video:input video for clas, support single video , internet url, folder path containing series of videos
        Returns:
            list[dict:{videoname: "",class_ids: [], scores: [], label_names: []}],if label name path is None,label names will be empty
        """
        video_list = []
        assert isinstance(video, (str))

        # get input_tensor and output_tensor
        input_names = self.predictor.get_input_names()
        output_names = self.predictor.get_output_names()
        input_tensor_list = []
        output_tensor_list = []
        for item in input_names:
            input_tensor_list.append(self.predictor.get_input_handle(item))
        for item in output_names:
            output_tensor_list.append(self.predictor.get_output_handle(item))

        if isinstance(video, str):
            # download internet video
            if video.startswith('http'):
                if not os.path.exists(BASE_VIDEOS_DIR):
                    os.makedirs(BASE_VIDEOS_DIR)
                video_path = os.path.join(BASE_VIDEOS_DIR, 'tmp.mp4')
                download_with_progressbar(video, video_path)
                print("Current using video from Internet:{}, renamed as: {}".
                      format(video, video_path))
                video = video_path
            files = parse_file_paths(video)
        else:
            print('Please input legal video!')

        # Inferencing process
        InferenceHelper = ppTSM_Inference_helper(
            num_seg=self.args.num_seg,
            short_size=self.args.short_size,
            target_size=self.args.target_size,
            top_k=self.args.top_k)
        batch_num = self.args.batch_size
        for st_idx in range(0, len(files), batch_num):
            ed_idx = min(st_idx + batch_num, len(files))

            # Pre process batched input
            batched_inputs = InferenceHelper.preprocess_batch(
                files[st_idx:ed_idx])

            # run inference
            for i in range(len(input_tensor_list)):
                input_tensor_list[i].copy_from_cpu(batched_inputs[i])
            self.predictor.run()

            batched_outputs = []
            for j in range(len(output_tensor_list)):
                batched_outputs.append(output_tensor_list[j].copy_to_cpu())

            results_list = InferenceHelper.postprocess(batched_outputs,
                                                       print_output=False,
                                                       return_result=True)

            for res in results_list:
                classes = res["topk_class"]
                label_names = []
                if len(self.label_name_dict) != 0:
                    label_names = [self.label_name_dict[c] for c in classes]
                res["label_names"] = label_names

                print("Current video file: {0}".format(res["video_id"]))
                print("\ttop-{0} classes: {1}".format(len(res["topk_class"]),
                                                      res["topk_class"]))
                print("\ttop-{0} scores: {1}".format(len(res["topk_scores"]),
                                                     res["topk_scores"]))
                print("\ttop-{0} label names: {1}".format(
                    len(res["label_names"]), res["label_names"]))


def main():
    # for cmd
    args = parse_args(mMain=True)
    clas_engine = PaddleVideo(**(args.__dict__))
    clas_engine.predict(args.video_file)


if __name__ == '__main__':
    main()
