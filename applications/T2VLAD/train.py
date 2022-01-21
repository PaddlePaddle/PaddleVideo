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

import os
import time
import copy
import socket
import paddle
import argparse
import warnings

import numpy as np
import model.loss as module_loss
import model.model as module_arch
import model.metric as module_metric
import data_loader.data_loaders as module_data


from pathlib import Path
from utils import set_seeds
from trainer import Trainer
from test import evaluation
from mergedeep import merge, Strategy
from parse_config import ConfigParser
from logger.log_parser import log_summary
from utils import compute_dims, compute_trn_config

def run_exp(config):
    warnings.filterwarnings('ignore')
    logger = config.get_logger('train')

    expert_dims, raw_input_dims = compute_dims(config, logger)
    trn_config = compute_trn_config(config)

    if config._args.group_seed:
        seeds = [int(config._args.group_seed)]
    else:
        seeds = [int(x) for x in config._args.seeds.split(",")]

    for ii, seed in enumerate(seeds):
        tic = time.time()
        logger.info(f"{ii + 1}/{len(seeds)} Setting experiment random seed to {seed}")
        set_seeds(seed)
        config["seed"] = seed

        model = config.init(
            name='arch',
            module=module_arch,
            expert_dims=expert_dims,
            text_dim=config["experts"]["text_dim"],
            ce_shared_dim=config["experts"].get("ce_shared_dim", None),
            feat_aggregation=config["data_loader"]["args"]["feat_aggregation"],
        )
        logger.info(model)

        data_loaders = config.init(
            name='data_loader',
            module=module_data,
            logger=logger,
            raw_input_dims=raw_input_dims,
            text_feat=config["experts"]["text_feat"],
            text_dim=config["experts"]["text_dim"],
            text_agg=config["experts"]["text_agg"],
            use_zeros_for_missing=config["experts"].get("use_zeros_for_missing", False),
            eval_only=False,
        )

        loss = config.init(name="loss", module=module_loss)
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        lr_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.0001, step_size=5, gamma=0.9)
        optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler, weight_decay=1e-4, parameters=model.parameters(), grad_clip=paddle.nn.ClipGradByGlobalNorm(2))

        trainer = Trainer(
            model,
            loss,
            metrics,
            optimizer,
            config=config,
            data_loaders=data_loaders,
            lr_scheduler=lr_scheduler,
            mini_train=config._args.mini_train,
            visualizer=None,
            val_freq=config["trainer"].get("val_freq", 1),
            force_cpu_val=config.get("force_cpu_val", False),
            skip_first_n_saves=config["trainer"].get("skip_first_n_saves", 0),
            include_optim_in_save_model=config["trainer"].get("include_optim_in_save_model", 1),
            cache_targets=set(config.get("cache_targets", [])),
        )
        trainer.train()
        best_model_path = config.save_dir / "trained_model.pdparams"
        duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - tic))
        logger.info(f"Training took {duration}")

    # If multiple runs were conducted, report relevant statistics
    if len(seeds) > 1:
        log_summary(
            logger=logger,
            log_path=config.log_path,
            eval_mode=config["eval_mode"],
            fixed_num_epochs=config["trainer"]["epochs"],
        )
    print(f"Log file stored at {config.log_path}")

    # Report the location of the "best" model of the final seeded run (here
    # "best" corresponds to the model with the highest geometric mean over the
    # R@1, R@5 and R@10 metrics when a validation set is used, or simply the final
    # epoch of training for fixed-length schedules).
    print(f"The best performing model can be found at {str(best_model_path)}")


def main():
    args = argparse.ArgumentParser(description='Main entry point for training')
    args.add_argument('--config', help='config file path')
    args.add_argument('--resume', help='path to latest model (default: None)')
    args.add_argument('--mini_train', action="store_true")
    args.add_argument('--group_id', help="if supplied, group these experiments")
    args.add_argument('--disable_workers', action="store_true")
    args.add_argument('--refresh_lru_cache', action="store_true")
    args.add_argument('--train_single_epoch', action="store_true")
    args.add_argument('--purge_exp_dir', action="store_true",
                      help="remove all previous experiments with the given config")
    args.add_argument("--dbg", default="ipdb.set_trace")
    args.add_argument("--custom_args", help="qualified key,val pairs")

    # Seeds can either be passed directly as a comma separated list at the command line,
    # or individually for separate experiments as a group (used for slurm experiments)
    seed_args = args.add_mutually_exclusive_group()
    seed_args.add_argument('--seeds', default="0", help="comma separated list of seeds")
    seed_args.add_argument('--group_seed', help="seed for group member")
    args = ConfigParser(args)
    os.environ["PYTHONBREAKPOINT"] = args._args.dbg
    args["data_loader"]["args"]["refresh_lru_cache"] = args._args.refresh_lru_cache
    msg = (f"Expected the number of training epochs ({args['trainer']['epochs']})"
           f"to exceed the save period ({args['trainer']['save_period']}), otherwise"
           " no checkpoints will be saved.")
    assert args["trainer"]["epochs"] >= args["trainer"]["save_period"], msg
    run_exp(config=args)


if __name__ == '__main__':
    main()
