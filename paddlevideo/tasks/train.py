# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import time
import os
import os.path as osp

import paddle
from ..loader import build_dataloader
from ..solver import build_lr, build_optimizer
from ..utils import do_preciseBN
from paddlevideo.utils import get_logger, coloring
from paddlevideo.utils import (AverageMeter, build_metric, log_batch, log_epoch,
                               save, mkdir)


def train_model(model, dataset, cfg, parallel=True, validate=True):
    """Train model entry

    Args:
    	model (paddle.nn.Layer): The model to be trained.
   	dataset (paddle.dataset): Train dataset.
    	cfg (dict): configuration.
        validate (bool): Whether to do evaluation. Default: False.

    """
    logger = get_logger("paddlevideo")
    batch_size = cfg.DATASET.get('batch_size', 2)
    places = paddle.set_device('gpu')

    train_dataset = dataset[0]
    train_dataloader_setting = dict(
        batch_size=batch_size,
        # default num worker: 0, which means no subprocess will be created
        num_workers=cfg.DATASET.get('num_workers', 0),
        places=places)
    train_loader = build_dataloader(train_dataset, **train_dataloader_setting)

    if validate:
        valid_dataset = dataset[1]
        validate_dataloader_setting = dict(batch_size=batch_size,
                                           num_workers=cfg.DATASET.get(
                                               'num_workers', 0),
                                           places=places,
                                           drop_last=False,
                                           shuffle=False)
        valid_loader = build_dataloader(valid_dataset,
                                        **validate_dataloader_setting)

    lr = build_lr(cfg.OPTIMIZER.learning_rate)
    optimizer = build_optimizer(cfg.OPTIMIZER,
                                lr,
                                parameter_list=model.parameters())

    if parallel:
        model = paddle.DataParallel(model)
    best = 0.
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        metric_list = build_metric()
        tic = time.time()
        for i, data in enumerate(train_loader):
            metric_list['reader_time'].update(time.time() - tic)
            if parallel:
                outputs = model._layers.train_step(data)
            else:
                outputs = model.train_step(data)

            avg_loss = outputs['loss']
            avg_loss.backward()

            optimizer.step()
            optimizer.clear_grad()

            # log metric
            metric_list['lr'].update(
                optimizer._global_learning_rate().numpy()[0], batch_size)
            for name, value in outputs.items():
                metric_list[name].update(value.numpy()[0], batch_size)
            metric_list['batch_time'].update(time.time() - tic)
            tic = time.time()

            if i % cfg.get("log_interval", 10) == 0:
                ips = "ips: {:.5f} instance/sec.".format(
                    batch_size / metric_list["batch_time"].val)
                log_batch(metric_list, i, epoch, cfg.epochs, "train", ips)
        # learning scheduler step
        lr.step()

        ips = "ips: {:.5f} instance/sec.".format(
            batch_size * metric_list["batch_time"].count /
            metric_list["batch_time"].sum)
        log_epoch(metric_list, epoch, "train", ips)

        def evaluate(best):
            model.eval()
            metric_list = build_metric()
            tic = time.time()
            for i, data in enumerate(valid_loader):
                if parallel:
                    outputs = model._layers.val_step(data)
                else:
                    outputs = model.val_step(data)

                # log_metric
                for name, value in outputs.items():
                    metric_list[name].update(value.numpy()[0], batch_size)
                metric_list['batch_time'].update(time.time() - tic)
                tic = time.time()

                if i % cfg.get("log_interval", 10) == 0:
                    ips = "ips: {:.5f} instance/sec.".format(
                        batch_size / metric_list["batch_time"].val)
                    log_batch(metric_list, i, epoch, cfg.epochs, "val", ips)
            ips = "ips: {:.5f} instance/sec.".format(
                batch_size * metric_list["batch_time"].count /
                metric_list["batch_time"].sum)
            log_epoch(metric_list, epoch, "val", ips)

            if metric_list['top1'].avg > best:
                best = metric_list['top1'].avg
            return best

        model_name = cfg.model_name
        output_dir = cfg.get("output_dir", f"./output/{model_name}")
        mkdir(output_dir)
        opt_state_dict = optimizer.state_dict()
        opt_name = cfg['OPTIMIZER']['name']

        if cfg.get("PRECISEBN") and (epoch -
                                     1) % cfg.PRECISEBN.preciseBN_interval == 0:
            do_preciseBN(
                model, train_loader, parallel,
                min(cfg.PRECISEBN.num_iters_preciseBN, len(train_loader)))
        if validate:
            with paddle.fluid.dygraph.no_grad():
                best = evaluate(best)

            # save best
            save(opt_state_dict, osp.join(output_dir, f"{opt_name}.pdopt"))
            save(model.state_dict(),
                 osp.join(output_dir, model_name + "_best.pdparams"))
            best = int(best * 10000) / 10000
            logger.info(
                f"Already save the best model (top1 acc){best} weights and optimizer params in epoch {epoch}"
            )

        if not validate and epoch % cfg.get("save_interval", 10) == 0:
            save(opt_state_dict, osp.join(output_dir, f"{opt_name}.pdopt"))
            save(model.state_dict(),
                 osp.join(output_dir, model_name + f"_epoch_{epoch}.pdparams"))

    logger.info(f'training {cfg.model_name} finished')
