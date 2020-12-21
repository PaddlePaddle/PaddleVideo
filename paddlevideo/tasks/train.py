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

import time
import os.path as osp

import paddle
from ..loader import build_dataloader
from ..solver import build_lr, build_optimizer
from ..utils import do_preciseBN
from paddlevideo.utils import get_logger, coloring
from paddlevideo.utils import (AverageMeter, build_record, log_batch, log_epoch,
                               save, load, mkdir)


def train_model(model, dataset, cfg, parallel=True, validate=True):
    """Train model entry

    Args:
    	model (paddle.nn.Layer): The model to be trained.
   	dataset (paddle.dataset): Train dataset.
    	cfg (dict): configuration.
        validate (bool): Whether to do evaluation. Default: False.

    """
    logger = get_logger("paddlevideo")
    #single card batch size
    batch_size = cfg.DATASET.get('batch_size', 8)
    places = paddle.set_device('gpu')

    num_workers = cfg.DATASET.get('num_workers', 0)
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    mkdir(output_dir)

    # 1. Construct dataloader
    train_dataset = dataset[0]
    train_dataloader_setting = dict(
        batch_size=batch_size,
        # default num worker: 0, which means no subprocess will be created
        num_workers=num_workers,
        collate_fn_cfg=cfg.get('MIX', None),
        places=places)

    train_loader = build_dataloader(train_dataset, **train_dataloader_setting)
    if validate:
        valid_dataset = dataset[1]
        validate_dataloader_setting = dict(batch_size=batch_size,
                                           num_workers=num_workers,
                                           places=places,
                                           drop_last=False,
                                           shuffle=False)
        valid_loader = build_dataloader(valid_dataset,
                                        **validate_dataloader_setting)

    # 2. Construct optimizer
    lr = build_lr(cfg.OPTIMIZER.learning_rate, len(train_loader))
    optimizer = build_optimizer(cfg.OPTIMIZER,
                                lr,
                                parameter_list=model.parameters())

    # used for multi-card
    if parallel:
        model = paddle.DataParallel(model)

    # Resume
    resume_epoch = cfg.get("resume_epoch", 0)
    if resume_epoch:
        filename = osp.join(output_dir,
                            model_name + f"_epoch_{resume_epoch:05d}")
        resume_model_dict = load(filename + '.pdparams')
        resume_opt_dict = load(filename + '.pdopt')
        model.set_state_dict(resume_model_dict)
        optimizer.set_state_dict(resume_opt_dict)

    # 3. Train Model
    best = 0.
    for epoch in range(1, cfg.epochs + 1):
        if epoch <= resume_epoch:
            logger.info(
                f"| epoch: [{epoch}] <= resume_epoch: [{ resume_epoch}], continue... "
            )
            continue
        model.train()
        record_list = build_record(cfg.MODEL)
        tic = time.time()
        for i, data in enumerate(train_loader):
            record_list['reader_time'].update(time.time() - tic)
            # 3.1 forward
            if parallel:
                outputs = model._layers.train_step(data)
            else:
                outputs = model.train_step(data)
            #3.2 backward
            avg_loss = outputs['loss']
            avg_loss.backward()
            #3.3 minimize
            optimizer.step()
            optimizer.clear_grad()

            # log record
            record_list['lr'].update(
                optimizer._global_learning_rate().numpy()[0], batch_size)
            for name, value in outputs.items():
                record_list[name].update(value.numpy()[0], batch_size)
            record_list['batch_time'].update(time.time() - tic)
            tic = time.time()

            if i % cfg.get("log_interval", 10) == 0:
                ips = "ips: {:.5f} instance/sec.".format(
                    batch_size / record_list["batch_time"].val)
                log_batch(record_list, i, epoch, cfg.epochs, "train", ips)

            # learning rate iter step
            if cfg.OPTIMIZER.learning_rate.get("iter_step"):
                lr.step()

        # learning rate epoch step
        if not cfg.OPTIMIZER.learning_rate.get("iter_step"):
            lr.step()

        ips = "ips: {:.5f} instance/sec.".format(
            batch_size * record_list["batch_time"].count /
            record_list["batch_time"].sum)
        log_epoch(record_list, epoch, "train", ips)

        def evaluate(best):
            model.eval()
            record_list = build_record(cfg.MODEL)
            record_list.pop('lr')
            tic = time.time()
            for i, data in enumerate(valid_loader):

                if parallel:
                    outputs = model._layers.val_step(data)
                else:
                    outputs = model.val_step(data)

                # log_record
                for name, value in outputs.items():
                    record_list[name].update(value.numpy()[0], batch_size)

                record_list['batch_time'].update(time.time() - tic)
                tic = time.time()

                if i % cfg.get("log_interval", 10) == 0:
                    ips = "ips: {:.5f} instance/sec.".format(
                        batch_size / record_list["batch_time"].val)
                    log_batch(record_list, i, epoch, cfg.epochs, "val", ips)

            ips = "ips: {:.5f} instance/sec.".format(
                batch_size * record_list["batch_time"].count /
                record_list["batch_time"].sum)
            log_epoch(record_list, epoch, "val", ips)

            best_flag = False
            if record_list.get('top1') and record_list['top1'].avg > best:
                best = record_list['top1'].avg
                best_flag = True
            return best, best_flag

        # use precise bn to improve acc
        if cfg.get("PRECISEBN") and (
            (epoch - 1) % cfg.PRECISEBN.preciseBN_interval == 0
                or epoch == cfg.epochs):
            do_preciseBN(
                model, train_loader, parallel,
                min(cfg.PRECISEBN.num_iters_preciseBN, len(train_loader)))

        # 4. Validation
        if validate and (epoch - 1) % cfg.get("val_interval",
                                              1) == 0 or epoch == cfg.epochs:
            with paddle.fluid.dygraph.no_grad():
                best, save_best_flag = evaluate(best)
            # save best
            if save_best_flag:
                save(optimizer.state_dict(),
                     osp.join(output_dir, model_name + "_best.pdopt"))
                save(model.state_dict(),
                     osp.join(output_dir, model_name + "_best.pdparams"))
                logger.info(
                    f"Already save the best model (top1 acc){int(best * 10000) / 10000}"
                )

        # 5. Save model and optimizer
        if (epoch - 1) % cfg.get("save_interval",
                                 10) == 0 or epoch == cfg.epochs:
            save(optimizer.state_dict(),
                 osp.join(output_dir, model_name + f"_epoch_{epoch:05d}.pdopt"))
            save(
                model.state_dict(),
                osp.join(output_dir,
                         model_name + f"_epoch_{epoch:05d}.pdparams"))

    logger.info(f'training {model_name} finished')
