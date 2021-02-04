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
from ..loader.builder import build_dataloader, build_dataset
from ..modeling.builder import build_model
from ..solver import build_lr, build_optimizer
from ..utils import do_preciseBN
from paddlevideo.utils import get_logger, coloring
from paddlevideo.utils import (AverageMeter, build_record, log_batch, log_epoch,
                               save, load, mkdir)


def train_model(cfg, weights=None, parallel=True, validate=True):
    """Train model entry

    Args:
    	cfg (dict): configuration.
        weights (str): weights path for finetuning.
    	parallel (bool): Whether multi-cards training. Default: True.
        validate (bool): Whether to do evaluation. Default: False.

    """

    logger = get_logger("paddlevideo")
    batch_size = cfg.DATASET.get('batch_size', 8)
    valid_batch_size = cfg.DATASET.get('valid_batch_size', batch_size)
    places = paddle.set_device('gpu')

    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    mkdir(output_dir)

    # 1. Construct model
    model = build_model(cfg.MODEL)
    if parallel:
        model = paddle.DataParallel(model)

    # 2. Construct dataset and dataloader
    train_dataset = build_dataset((cfg.DATASET.train, cfg.PIPELINE.train))
    train_dataloader_setting = dict(batch_size=batch_size,
                                    num_workers=num_workers,
                                    collate_fn_cfg=cfg.get('MIX', None),
                                    places=places)

    train_loader = build_dataloader(train_dataset, **train_dataloader_setting)
    if validate:
        valid_dataset = build_dataset((cfg.DATASET.valid, cfg.PIPELINE.valid))
        validate_dataloader_setting = dict(
            batch_size=valid_batch_size,
            num_workers=num_workers,
            places=places,
            drop_last=False,
            shuffle=cfg.DATASET.get(
                'shuffle_valid',
                False)  #NOTE: attention lstm need shuffle valid data.
        )
        valid_loader = build_dataloader(valid_dataset,
                                        **validate_dataloader_setting)

    # 3. Construct solver.
    lr = build_lr(cfg.OPTIMIZER.learning_rate, len(train_loader))
    optimizer = build_optimizer(cfg.OPTIMIZER,
                                lr,
                                parameter_list=model.parameters())

    # Resume
    resume_epoch = cfg.get("resume_epoch", 0)
    if resume_epoch:
        filename = osp.join(output_dir,
                            model_name + f"_epoch_{resume_epoch:05d}")
        resume_model_dict = load(filename + '.pdparams')
        resume_opt_dict = load(filename + '.pdopt')
        model.set_state_dict(resume_model_dict)
        optimizer.set_state_dict(resume_opt_dict)

    # Finetune:
    if weights:
        assert resume_epoch == 0, f"Conflict occurs when finetuning, please switch resume function off by setting resume_epoch to 0 or not indicating it."
        model_dict = load(weights)
        model.set_state_dict(model_dict)

    # 4. Train Model
    best = 0.
    for epoch in range(0, cfg.epochs):
        if epoch < resume_epoch:
            logger.info(
                f"| epoch: [{epoch+1}] <= resume_epoch: [{ resume_epoch}], continue... "
            )
            continue
        model.train()
        record_list = build_record(cfg.MODEL)
        tic = time.time()
        for i, data in enumerate(train_loader):
            record_list['reader_time'].update(time.time() - tic)
            # 4.1 forward
            if parallel:
                outputs = model._layers.train_step(data)
            else:
                outputs = model.train_step(data)
            # 4.2 backward
            avg_loss = outputs['loss']
            avg_loss.backward()
            # 4.3 minimize
            optimizer.step()
            optimizer.clear_grad()

            # log record
            record_list['lr'].update(optimizer._global_learning_rate(),
                                     batch_size)
            for name, value in outputs.items():
                record_list[name].update(value, batch_size)

            record_list['batch_time'].update(time.time() - tic)
            tic = time.time()

            if i % cfg.get("log_interval", 10) == 0:
                ips = "ips: {:.5f} instance/sec.".format(
                    batch_size / record_list["batch_time"].val)
                log_batch(record_list, i, epoch + 1, cfg.epochs, "train", ips)

            # learning rate iter step
            if cfg.OPTIMIZER.learning_rate.get("iter_step"):
                lr.step()

        # learning rate epoch step
        if not cfg.OPTIMIZER.learning_rate.get("iter_step"):
            lr.step()

        ips = "ips: {:.5f} instance/sec.".format(
            batch_size * record_list["batch_time"].count /
            record_list["batch_time"].sum)
        log_epoch(record_list, epoch + 1, "train", ips)

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
                    record_list[name].update(value, batch_size)

                record_list['batch_time'].update(time.time() - tic)
                tic = time.time()

                if i % cfg.get("log_interval", 10) == 0:
                    ips = "ips: {:.5f} instance/sec.".format(
                        batch_size / record_list["batch_time"].val)
                    log_batch(record_list, i, epoch + 1, cfg.epochs, "val", ips)

            ips = "ips: {:.5f} instance/sec.".format(
                batch_size * record_list["batch_time"].count /
                record_list["batch_time"].sum)
            log_epoch(record_list, epoch + 1, "val", ips)

            best_flag = False
            for top_flag in ['hit_at_one', 'top1']:
                if record_list.get(
                        top_flag) and record_list[top_flag].avg > best:
                    best = record_list[top_flag].avg
                    best_flag = True

            return best, best_flag

        # use precise bn to improve acc
        if cfg.get("PRECISEBN") and (epoch % cfg.PRECISEBN.preciseBN_interval
                                     == 0 or epoch == cfg.epochs - 1):
            do_preciseBN(
                model, train_loader, parallel,
                min(cfg.PRECISEBN.num_iters_preciseBN, len(train_loader)))

        # 5. Validation
        if validate and epoch % cfg.get("val_interval",
                                        1) == 0 or epoch == cfg.epochs - 1:
            with paddle.fluid.dygraph.no_grad():
                best, save_best_flag = evaluate(best)
            # save best
            if save_best_flag:
                save(optimizer.state_dict(),
                     osp.join(output_dir, model_name + "_best.pdopt"))
                save(model.state_dict(),
                     osp.join(output_dir, model_name + "_best.pdparams"))
                if model_name == "AttentionLstm":
                    logger.info(
                        f"Already save the best model (hit_at_one){best}")
                else:
                    logger.info(
                        f"Already save the best model (top1 acc){int(best *10000)/10000}"
                    )

        # 6. Save model and optimizer
        if epoch % cfg.get("save_interval", 1) == 0 or epoch == cfg.epochs - 1:
            save(
                optimizer.state_dict(),
                osp.join(output_dir,
                         model_name + f"_epoch_{epoch+1:05d}.pdopt"))
            save(
                model.state_dict(),
                osp.join(output_dir,
                         model_name + f"_epoch_{epoch+1:05d}.pdparams"))

    logger.info(f'training {model_name} finished')
