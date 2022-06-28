# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import copy
import random
import paddle
import logging
import argparse

import numpy as np
import model.model as module_arch
import model.metric as module_metric
import data_loader.data_loaders as module_data

from typing import Tuple
from pathlib import Path
from typeguard import typechecked
from mergedeep import Strategy, merge
from parse_config import ConfigParser
from trainer.trainer import verbose, ctxt_mgr
from utils.util import compute_dims, compute_trn_config

@typechecked
def compress_predictions(query_masks: np.ndarray, sims: np.ndarray, topk: int = 10):
    """We store the indices of the top-k predictions, rather than the full similarity
    matrix, to reduce storage requirements.

    NOTE: The similarity matrix contains `num_queries x num_videos` elements, where
    `num_queries = num_videos x max_num_queries_per_video`.  We first mask out
    locations in the similarity matrix that correspond to invalid queries (these are
    produced by videos with fewer than `max_num_queries_per_video` descriptions).
    """

    # validate the input shapes
    assert query_masks.ndim == 2, "Expected query_masks to be a matrix"
    query_num_videos, query_max_per_video = query_masks.shape
    sims_queries, sims_num_videos = sims.shape
    msg = (f"Expected sims and query masks to represent the same number of videos "
           f"(found {sims_num_videos} v {query_num_videos}")
    assert query_num_videos == sims_num_videos, msg
    msg = (f"Expected sims and query masks to represent the same number of queries "
           f"(found {sims_queries} v {query_num_videos * query_max_per_video}")
    assert query_max_per_video * query_num_videos == sims_queries, msg

    valid_sims = sims[query_masks.flatten().astype(np.bool)]
    ranks = np.argsort(-valid_sims, axis=1)
    return ranks[:, :topk]


@typechecked
def get_model_and_data_loaders(
        config: ConfigParser,
        logger: logging.Logger,
        model_path: Path,
) -> Tuple[paddle.nn.Layer, module_data.ExpertDataLoader]:
    expert_dims, raw_input_dims = compute_dims(config)
    trn_config = compute_trn_config(config)

    data_loaders = config.init(
        name='data_loader',
        module=module_data,
        logger=logger,
        raw_input_dims=raw_input_dims,
        text_feat=config["experts"]["text_feat"],
        text_dim=config["experts"]["text_dim"],
        text_agg=config["experts"]["text_agg"],
        use_zeros_for_missing=config["experts"].get("use_zeros_for_missing", False),
        eval_only=True,
    )

    model = config.init(
        name='arch',
        module=module_arch,
        expert_dims=expert_dims,
        text_dim=config["experts"]["text_dim"],
        ce_shared_dim=config["experts"].get("ce_shared_dim", None),
        feat_aggregation=config["data_loader"]["args"]["feat_aggregation"],
    )
    model_path = config._args.resume
    logger.info(f"Loading checkpoint: {model_path} ...")
    checkpoint = paddle.load(model_path)
    state_dict = checkpoint
    if config['n_gpu'] > 1:
        model = paddle.DataParallel(model)
    model.load_dict(state_dict)

    return model, data_loaders


def evaluation(config, logger=None, trainer=None):

    if logger is None:
        logger = config.get_logger('test')

    if getattr(config._args, "eval_from_training_config", False):
        eval_conf = copy.deepcopy(config)
        merge(eval_conf._config, config["eval_settings"], strategy=Strategy.REPLACE)
        config = eval_conf

    logger.info("Running evaluation with configuration:")
    logger.info(config)

    # Set the random initial seeds
    seed = config["seed"]
    logger.info(f"Setting experiment random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    model, data_loaders = get_model_and_data_loaders(
        config=config,
        logger=logger,
        model_path=Path(config._args.resume),
    )
    logger.info(model)

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # prepare model for testing.  Note that some datasets fail to fit the retrieval
    # set on the GPU, so we run them on the CPU
    model.eval()

    with paddle.no_grad():
        samples, meta = data_loaders["retrieval"]
        #import pdb; pdb.set_trace()
        # To use the nan-checks safely, we need make temporary copies of the data
        all_text_num = samples['text'].shape[0]
        text_keys = ['text', 'cap_id', 'att_mask', 'text_token_mask']
        chk = 100
        tck = 100 

        if samples['text'].shape[0] % chk == 0:
            vid_batch = samples['text'].shape[0] // chk
        else:
            vid_batch = samples['text'].shape[0] // chk + 1
        if samples['text'].shape[0] % tck == 0:
            text_batch  =  samples['text'].shape[0] // tck
        else: 
            text_batch  =  samples['text'].shape[0] // tck + 1
        sub_sims = []
        for idx in range(text_batch):
            if idx % 5 == 0:
                print(idx,'/',text_batch)
            sub_samples = {}
            for key in text_keys:
                sub_samples.update({key: samples[key][idx*tck:idx*tck+tck]})
            subsub_sims = []
            for vid in range(vid_batch):
                sub_samples['experts'] = {}
                sub_samples['ind'] = {}
                for expert in samples['experts'].keys():
                    sub_samples['experts'][expert] = samples['experts'][expert][vid*chk:vid*chk+chk]
                    sub_samples['ind'][expert] = samples['ind'][expert][vid*chk:vid*chk+chk]
                with ctxt_mgr(sub_samples) as valid:
                    output = model(**valid)
                subsub_sims.append(output["cross_view_conf_matrix"].cpu())
            subsub_sims = paddle.concat(subsub_sims, axis=1)
            sub_sims.append(subsub_sims)
        sub_sims = paddle.concat(sub_sims, axis=0)
        sims = paddle.to_tensor(sub_sims, dtype='float32').numpy()
        dataset = data_loaders.dataset_name

        nested_metrics = {}
        for metric in metrics:
            metric_name = metric.__name__
            res = metric(sims, query_masks=meta["query_masks"])
            verbose(epoch=0, metrics=res, name=dataset, mode=metric_name)
            if trainer is not None:
                if not trainer.mini_train:
                    trainer.writer.set_step(step=0, mode="val")
                # avoid tensboard folding by prefixing
                metric_name_ = f"test_{metric_name}"
                trainer.log_metrics(res, metric_name=metric_name_, mode="val")
            nested_metrics[metric_name] = res

    log = {}
    for subkey, subval in nested_metrics.items():
        for subsubkey, subsubval in subval.items():
            log[f"test_{subkey}_{subsubkey}"] = subsubval
    for key, value in log.items():
        logger.info(" {:15s}: {}".format(str(key), value))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--config', default=None, type=str, help="config file path")
    args.add_argument('--resume', default=None, help='path to checkpoint for evaluation')
    args.add_argument('--eval_from_training_config', action="store_true",
                      help="if true, evaluate directly from a training config file.")
    args.add_argument("--custom_args", help="qualified key,val pairs")
    eval_config = ConfigParser(args)

    cfg_msg = "For evaluation, a model checkpoint must be specified via the --resume flag"
    assert eval_config._args.resume, cfg_msg
    if eval_config._config.get("eval_settings", False):
        merge(eval_config._config, eval_config["eval_settings"], strategy=Strategy.REPLACE)
        evaluation(eval_config)
