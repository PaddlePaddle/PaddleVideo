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

sys.path.append(
    "/workspace/bianjiang03/App_TableTennis/PaddleVideo/FootballAction/predict/action_detect"
)
import models.bmn_infer as prop_model
from utils.preprocess import get_images
from utils.config_utils import parse_config, print_configs
import utils.config_utils as config_utils

import logger

logger = logger.Logger()


def load_model(cfg_file="configs/configs.yaml"):
    """
    load_model
    """
    logger.info("load model ... ")
    global infer_configs
    infer_configs = parse_config(cfg_file)
    print_configs(infer_configs, "Infer")

    t0 = time.time()
    global prop_model
    prop_model = prop_model.InferModel(infer_configs)
    t1 = time.time()
    logger.info("step0: load model time: {} min\n".format((t1 - t0) * 1.0 / 60))


def video_classify(video_name, dataset_dir):
    """
    extract_feature
    """
    logger.info('predict ... ')
    logger.info(video_name)

    # step 1: extract feature

    feature_path = dataset_dir + video_name
    video_features = pickle.load(open(feature_path, 'rb'))
    print('===video_features===', video_name)

    # step2: get proposal
    t0 = time.time()
    bmn_results = prop_model.predict(infer_configs, material=video_features)
    t1 = time.time()
    logger.info(np.array(bmn_results).shape)
    logger.info("step2: proposal time: {} min".format((t1 - t0) * 1.0 / 60))

    return bmn_results


if __name__ == '__main__':
    dataset_dir = '/workspace/bianjiang03/DATA/Features_competition_test_A/'
    output_dir = '/workspace/bianjiang03/DATA'
    if not os.path.exists(output_dir + '/Output_for_bmn'):
        os.mkdir(output_dir + '/Output_for_bmn')
    results = []

    load_model()

    directory = os.fsencode(dataset_dir)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        bmn_results = video_classify(filename, dataset_dir)
        results.append({
            'video_name': filename.split('.pkl')[0],
            'num_proposal': len(bmn_results),
            'bmn_results': bmn_results
        })

    with open(output_dir + '/Output_for_bmn/prop.json', 'w',
              encoding='utf-8') as f:
        data = json.dumps(results, indent=4, ensure_ascii=False)
        f.write(data)

    print('Done with the inference!')
