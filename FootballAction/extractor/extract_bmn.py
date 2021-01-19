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
    global image_model, audio_model, prop_model, classify_model
    prop_model = net_prop.ModelProp(infer_configs, "BMN")
    # classify_model = net_prop.ModelProp(infer_configs, "ACTION")
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

    feature_path = video_name.replace(".mp4", ".pkl").replace("mp4", "features")
    infer_configs['COMMON']['feature_path'] = feature_path

    # step2: get proposal
    t0 = time.time()
    bmn_results = prop_model.predict_BMN(infer_configs)
    t1 = time.time()
    logger.info(np.array(bmn_results).shape)
    logger.info("step2: proposal time: {} min".format((t1 - t0) * 1.0 / 60))

    return bmn_results


if __name__ == '__main__':
    dataset_dir = "/home/work/datasets/EuroCup2016"
    if not os.path.exists(dataset_dir + '/feature_bmn'):
        os.mkdir(dataset_dir + '/feature_bmn')
    results = []

    load_model()

    video_url = os.path.join(dataset_dir, 'url.list')
    with open(video_url, 'r') as f:
        lines = f.readlines()
    lines = [os.path.join(dataset_dir, k.strip()) for k in lines]

    for line in lines:
        bmn_results = video_classify(line)
        results.append({
            'video_name': os.path.basename(line).split('.')[0],
            'num_proposal': len(bmn_results),
            'bmn_results': bmn_results
        })

    with open(dataset_dir + '/feature_bmn/prop.json', 'w',
              encoding='utf-8') as f:
        data = json.dumps(results, indent=4, ensure_ascii=False)
        f.write(data)
