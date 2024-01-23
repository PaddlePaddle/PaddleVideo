#!/usr/bin/env python
# coding=utf-8
"""
infer model
"""
import sys
import os
import numpy as np
import json
import pickle
import argparse
import time

import numpy as np
import paddle

from datareader import get_reader
from config import merge_configs, parse_config, print_configs


def parse_args():
    """parse_args
    """
    parser = argparse.ArgumentParser("Paddle Video infer script")
    parser.add_argument('--model_name',
                        type=str,
                        default='BaiduNet',
                        help='name of model to train.')
    parser.add_argument('--config',
                        type=str,
                        default='configs/conf.txt',
                        help='path to config file of model')
    parser.add_argument('--output', type=str, default=None, help='output path')
    parser.add_argument('--use_gpu',
                        type=bool,
                        default=True,
                        help='default use gpu.')
    parser.add_argument('--save_inference_model',
                        type=str,
                        default=None,
                        help='save inference path')
    args = parser.parse_args()
    return args

class InferModel(object):
    """lstm infer"""
    def __init__(self, cfg, name='ACTION'): 
        name = name.upper()
        self.name           = name
        self.threshold      = cfg.INFER.threshold
        self.cfg            = cfg
        self.label_map      = load_class_file(cfg.MODEL.class_name_file)
       

    def load_inference_model(self, model_dir, use_gpu=True):
        """model_init
        """
        model_file = os.path.join(model_dir, "model")
        params_file = os.path.join(model_dir, "params")
        config = paddle.inference.Config(model_file, params_file)
        if use_gpu:
            config.enable_use_gpu(1024)
        else:
            config.disable_gpu()
        config.switch_ir_optim(True)  # default true
        config.enable_memory_optim()
        # use zero copy
        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config)
        # build input tensor and output tensor
        self.build_input_output()

    def build_input_output(self):
        """build_input_output
        """
        input_names = self.predictor.get_input_names()
        # input
        self.input_rgb_tensor = self.predictor.get_input_handle(input_names[0])
        self.input_audio_tensor = self.predictor.get_input_handle(input_names[1])
        self.input_text_tensor = self.predictor.get_input_handle(input_names[2])

        # output
        output_names = self.predictor.get_output_names()
        self.output_tensor = self.predictor.get_output_handle(output_names[0])

    def preprocess_for_lod_data(self, input):
        """pre process"""
        input_arr = []
        input_lod = [0]
        start_lod = 0
        end_lod = 0
        for sub_item in input:
            end_lod = start_lod + len(sub_item)
            input_lod.append(end_lod)
            input_arr.extend(sub_item)
            start_lod = end_lod
        input_arr = np.array(input_arr)
        return input_arr, [input_lod]

    def predict(self):
        """predict"""
        infer_reader = get_reader(self.name, 'infer', self.cfg)
        probs = []
        video_ids = []
        label_map_inverse = {value: key for key, value in self.label_map.items()}
        for infer_iter, data in enumerate(infer_reader()):
            # video_id = [[items[-2], items[-1]] for items in data]
            rgb = [items[0] for items in data]
            audio = [items[1] for items in data]
            text = np.array([items[2] for items in data])
            videos = np.array([items[3] for items in data])

            rgb_arr, rgb_lod = self.preprocess_for_lod_data(rgb)
            audio_arr, audio_lod = self.preprocess_for_lod_data(audio)

            self.input_rgb_tensor.copy_from_cpu(rgb_arr.astype('float32'))
            self.input_rgb_tensor.set_lod(rgb_lod)

            self.input_audio_tensor.copy_from_cpu(audio_arr.astype('float32'))
            self.input_audio_tensor.set_lod(audio_lod)

            self.input_text_tensor.copy_from_cpu(text.astype('int64'))

            self.predictor.run()
            output = self.output_tensor.copy_to_cpu()
            probs.extend(list(output))
            video_ids.extend(videos)
        assert len(video_ids) == len(probs)
        result = []
        for video_id, prob in zip(video_ids, probs):
            label_idx = list(np.where(prob >= self.threshold)[0])
            result.append({
                "video_id": video_id,
                "labels": [
                    (label_map_inverse[str(idx)], float(prob[idx])) for idx in label_idx
                ]
            })
        return result


def load_class_file(class_file):
    """
    load_class_file
    """
    class_lines = open(class_file, 'r', encoding='utf8').readlines()
    class_dict = {}
    for i, line in enumerate(class_lines):
        tmp = line.strip().split('\t')
        word = tmp[0]
        index = str(i)
        if len(tmp) == 2:
            index = tmp[1]
        class_dict[word] = index
    return class_dict


def infer(args):
    """
    infer main
    """
    config = parse_config(args.config)
    infer_config = merge_configs(config, 'infer', vars(args))
    print_configs(infer_config, 'infer')
    infer_obj = InferModel(infer_config, name=args.model_name)
    infer_obj.load_inference_model(args.save_inference_model, use_gpu=args.use_gpu)
    rt = infer_obj.predict()
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(rt, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    args = parse_args()
    infer(args)
