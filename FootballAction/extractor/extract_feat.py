#!./python27-gcc482/bin/python
# coding: utf-8
"""
BAIDU CLOUD action
"""

import os
import sys
import pickle
import json
import time
import shutil

import numpy as np

sys.path.append("../predict")
import prop_net as net_prop
from utils.config_utils import parse_config, print_configs
import utils.config_utils as config_utils
import utils.extract_mp4 as extract_mp4

import logging
logger = logging.getLogger(__name__)


def load_model(cfg_file="configs/configs.yaml"):
    """
    load_model
    """
    logger.info("load model ... ")
    global infer_configs
    infer_configs = parse_config(cfg_file)
    print_configs(infer_configs, "Infer")

    t0 = time.time()
    global image_model, audio_model
    image_model = net_prop.ModelProp(infer_configs, "TSN")
    audio_model = net_prop.ModelProp(infer_configs, 'AUDIO')
    t1 = time.time()
    logger.info("step0: load model time: {} min\n".format((t1 - t0) * 1.0 / 60))


def video_classify(video_name):
    """
    extract_feature
    """
    logger.info('predict ... ')
    logger.info(video_name)
    imgs_path = video_name.replace(".mp4", "").replace("mp4", "frames")
    pcm_path = video_name.replace(".mp4", ".pcm").replace("mp4", "pcm")

    # step 1: extract feature
    t0 = time.time()
    image_path_list = extract_mp4.get_images(imgs_path)
    infer_configs['TSN']['frame_list'] = image_path_list
    infer_configs['AUDIO']['pcm_file'] = pcm_path
    image_features = image_model.predict_TSN(infer_configs)
    audio_features = audio_model.predict_audio(infer_configs)

    np_image_features = np.array(image_features, dtype=np.float32)
    np_audio_features = np.array(audio_features, dtype=np.float32)
    t1 = time.time()

    logger.info(np_image_features.shape, np_audio_features.shape)
    logger.info("step1: feature extract time: {} min".format(
        (t1 - t0) * 1.0 / 60))
    video_features = {
        'image_feature': np_image_features,
        'audio_feature': np_audio_features
    }

    # save feature
    feature_path = video_name.replace(".mp4", ".pkl").replace("mp4", "features")
    feat_pkl_str = pickle.dumps(video_features,
                                protocol=pickle.HIGHEST_PROTOCOL)
    with open(feature_path, 'wb') as fout:
        fout.write(feat_pkl_str)


if __name__ == '__main__':
    dataset_dir = "/home/work/datasets/EuroCup2016"
    if not os.path.exists(dataset_dir + '/features'):
        os.mkdir(dataset_dir + '/features')

    load_model()

    video_url = os.path.join(dataset_dir, 'url.list')
    with open(video_url, 'r') as f:
        lines = f.readlines()
    lines = [os.path.join(dataset_dir, k.strip()) for k in lines]

    for line in lines:
        video_classify(line)
