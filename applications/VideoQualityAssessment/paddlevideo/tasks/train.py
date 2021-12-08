"""
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
"""

import time
import os.path as osp

import paddle
import paddle.distributed.fleet as fleet
from ..loader.builder import build_dataloader, build_dataset
from ..modeling.builder import build_model
from ..solver import build_lr, build_optimizer
from ..metrics import build_metric
from ..utils import do_preciseBN
from paddlevideo.utils import get_logger, coloring
from paddlevideo.utils import (AverageMeter, build_rec_record, log_batch, log_epoch,
                               save, load, mkdir)
#from paddlevideo.metrics import QualityMetric
import numpy as np
from scipy import stats

def train_model(cfg,
                weights=None,
                parallel=True,
                validate=True,
                amp=False,
                fleet=False):
    """Train model entry

    Args:
    	cfg (dict): configuration.
        weights (str): weights path for finetuning.
    	parallel (bool): Whether multi-cards training. Default: True.
        validate (bool): Whether to do evaluation. Default: False.

    """
    if fleet:
        fleet.init(is_collective=True)

    logger = get_logger("paddlevideo")
    batch_size = cfg.DATASET.get('batch_size', 8)
    valid_batch_size = cfg.DATASET.get('valid_batch_size', batch_size)
    places = paddle.set_device('gpu')

    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", "./output/model_name/")
    mkdir(output_dir)

    # 1. Construct model
    model = build_model(cfg.MODEL)
    if parallel:
        model = paddle.DataParallel(model)

    if fleet:
        model = paddle.distributed_model(model)

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
    if fleet:
        optimizer = fleet.distributed_optimizer(optimizer)
    # Resume
    resume_epoch = cfg.get("resume_epoch", 0)
    if resume_epoch:
        filename = osp.join(output_dir,
                            model_name + "_epoch_{}".format(resume_epoch))
        resume_model_dict = load(filename + '.pdparams')
        resume_opt_dict = load(filename + '.pdopt')
        model.set_state_dict(resume_model_dict)
        optimizer.set_state_dict(resume_opt_dict)

    # Finetune:
    if weights:
        assert resume_epoch == 0, "Conflict occurs when finetuning, please switch resume function off by setting resume_epoch to 0 or not indicating it."
        model_dict = load(weights)
        model.set_state_dict(model_dict)

    # 4. Train Model
    ###AMP###
    if amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    best = 0.
    max_SROCC = 0
    max_PLCC = 0
    Metric = build_metric(cfg.METRIC)
    for epoch in range(0, cfg.epochs):
        if epoch < resume_epoch:
            logger.info(
                "| epoch: [{epoch+1}] <= resume_epoch: [{ resume_epoch}], continue... "
            )
            continue
        model.train()
        record_list = build_rec_record(cfg.MODEL)
        tic = time.time()
        train_output = []
        train_label = []
        
         
        for i, data in enumerate(train_loader):
            record_list['reader_time'].update(time.time() - tic)

            # 4.1 forward
            ###AMP###
            if amp:
                with paddle.amp.auto_cast(
                        custom_black_list={"temporal_shift", "reduce_mean"}):
                    if parallel:
                        outputs = model._layers.train_step(data)
                        ## required for DataParallel, will remove in next version
                        model._reducer.prepare_for_backward(
                            list(model._find_varbase(outputs)))
                    else:
                        outputs = model.train_step(data)

                train_output.extend(outputs['output'])
                train_label.extend(outputs['label'])
                    
                avg_loss = outputs['loss']
                scaled = scaler.scale(avg_loss)
                scaled.backward()
                # keep prior to 2.0 design
                scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()

            else:
                if parallel:
                    outputs = model._layers.train_step(data)
                    ## required for DataParallel, will remove in next version
                    model._reducer.prepare_for_backward(
                        list(model._find_varbase(outputs)))
                else:
                    outputs = model.train_step(data)
                
                train_output.extend(outputs['output'])
                train_label.extend(outputs['label'])
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
                if name == 'output' or name == 'label':
                    continue
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
        
        train_PLCC, train_SROCC = Metric.accumulate_train(train_output, train_label)
        logger.info("train_SROCC={}".format(train_SROCC))
        logger.info("train_PLCC={}".format(train_PLCC))

        ips = "ips: {:.5f} instance/sec.".format(
            batch_size * record_list["batch_time"].count /
            record_list["batch_time"].sum)
        log_epoch(record_list, epoch + 1, "train", ips)

        eval_output = []
        eval_label = []
        def evaluate(best, max_SROCC, max_PLCC):
            """evaluate"""
            model.eval()
            record_list = build_rec_record(cfg.MODEL)
            record_list.pop('lr')
            tic = time.time()
            
            for i, data in enumerate(valid_loader):

                if parallel:
                    outputs = model._layers.val_step(data)
                else:
                    outputs = model.val_step(data)
                eval_output.extend(outputs['output'])
                eval_label.extend(outputs['label'])

                # log_record
                for name, value in outputs.items():
                    if name == 'output' or name == 'label':
                        continue
                    record_list[name].update(value, batch_size)

                record_list['batch_time'].update(time.time() - tic)
                tic = time.time()

                if i % cfg.get("log_interval", 10) == 0:
                    ips = "ips: {:.5f} instance/sec.".format(
                        batch_size / record_list["batch_time"].val)
                    log_batch(record_list, i, epoch + 1, cfg.epochs, "val", ips)
            
            eval_PLCC, eval_SROCC = Metric.accumulate_train(eval_output, eval_label)
            logger.info("val_SROCC={}".format(eval_SROCC))
            logger.info("val_PLCC={}".format(eval_PLCC))
                  
            if max_SROCC <= eval_SROCC and max_PLCC <= eval_PLCC:
                max_SROCC = eval_SROCC
                max_PLCC = eval_PLCC
                logger.info("max_SROCC={}".format(max_SROCC))
                logger.info("max_PLCC={}".format(max_PLCC))
                save(optimizer.state_dict(),
                     osp.join(output_dir, model_name + "_best.pdopt"))
                save(model.state_dict(),
                     osp.join(output_dir, model_name + "_best.pdparams"))
            
            ips = "ips: {:.5f} instance/sec.".format(
                batch_size * record_list["batch_time"].count /
                record_list["batch_time"].sum)
            log_epoch(record_list, epoch + 1, "val", ips)

            return best, max_SROCC, max_PLCC

        # use precise bn to improve acc
        if cfg.get("PRECISEBN") and (epoch % cfg.PRECISEBN.preciseBN_interval
                                     == 0 or epoch == cfg.epochs - 1):
            do_preciseBN(
                model, train_loader, parallel,
                min(cfg.PRECISEBN.num_iters_preciseBN, len(train_loader)))

        # 5. Validation
        if validate and (epoch % cfg.get("val_interval", 1) == 0
                         or epoch == cfg.epochs - 1):
            with paddle.fluid.dygraph.no_grad():
                best, max_SROCC, max_PLCC = evaluate(best, max_SROCC, max_PLCC)

        # 6. Save model
        if epoch % cfg.get("save_interval", 1) == 0 or epoch == cfg.epochs - 1:
            save(optimizer.state_dict(), osp.join(output_dir, model_name + "_epoch_{}.pdopt".format(epoch)))
            save(model.state_dict(), osp.join(output_dir, model_name + "_epoch_{}.pdparams".format(epoch)))




    logger.info('training {model_name} finished')
