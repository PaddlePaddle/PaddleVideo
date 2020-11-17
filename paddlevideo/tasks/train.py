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
from collections import OrderedDict

import paddle
from ..loader import build_dataset, build_dataloader
from ..solver import build_lr, build_optimizer
from paddlevideo.utils import get_logger, coloring
from paddlevideo.utils import AverageMeter


def train_model(model,
                dataset,
                cfg,
                parallel=True,
	        validate=False):
    """Train model entry

    Args:
    	model (paddle.nn.Layer): The model to be trained.
   	dataset (paddle.dataset): Train dataset.
    	cfg (dict): configuration.
        validate (bool): Whether to do evaluation. Default: False.

    """

    logger = get_logger("paddlevideo")

    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    #build data loader, refer to the field ```dataset``` in the configuration for more details.
    batch_size = cfg.DATASET.get('batch_size', 2)
    dataloader_setting = dict(
        batch_size = batch_size,
        # default num worker: 0, which means no subprocess will be created
        num_workers = cfg.DATASET.get('num_workers', 1),
        places = paddle.CUDAPlace(paddle.distributed.ParallelEnv().dev_id) \
            if parallel else paddle.CUDAPlace(0)
        )

    data_loaders = [build_dataloader(ds, **dataloader_setting) for ds in dataset]

    #build optimizer, refer to the field ```optimizer``` in the configuration for more details.
    lr = build_lr(cfg.OPTIMIZER.learning_rate)
    optimizer = build_optimizer(cfg.OPTIMIZER, lr, parameter_list=model.parameters())


    if parallel:
        model = paddle.DataParallel(model)

    train_loader = data_loaders[0]
    
    metric_list = [
        ("loss", AverageMeter('loss', '7.5f')),
        ("lr", AverageMeter(
            'lr', 'f', need_avg=False)),
        ("top1", AverageMeter("top1", '.5f')),
        ("top5", AverageMeter("top5", '.5f')),
        ("batch_time", AverageMeter('elapse', '.5f')),
        ("reader_time", AverageMeter('reader ', '.5f')),
    ]
    metric_list = OrderedDict(metric_list)
    tic = time.time()


    for epoch in range(1, cfg.epochs + 1):
        metric_list['reader_time'].update(time.time() - tic)
        model.train()
        total_loss = 0.0
        total_sample = 0
        for i, data in enumerate(train_loader):
            if parallel:
                outputs = model._layers.train_step(data)
            else:
                outputs = model.train_step(data)

            avg_loss = outputs['loss']
            avg_loss.backward()

            optimizer.minimize(avg_loss)
            optimizer.step()
            optimizer.clear_grad()
            metric_list['lr'].update(
                optimizer._global_learning_rate().numpy()[0], batch_size)

            for name, value in outputs.items():
                metric_list[name].update(value.numpy()[0], batch_size)
            metric_list['batch_time'].update(time.time() - tic)
            tic = time.time()


            if i % cfg.get("log_interval", 10) == 0:

                fetchs_str = ' '.join([str(m.value) for m in metric_list.values()])
                epoch_str = "epoch:[{:>3d}/{:<3d}]".format(epoch, cfg.epochs)
                step_str = "{:s} step:{:<4d}".format("train", i)
                logger.info("{:s} {:s} {:s}s".format(
                    coloring(epoch_str, "HEADER")
                    if i == 0 else epoch_str,
                    coloring(step_str, "PURPLE"),
                    coloring(fetchs_str, 'OKGREEN')))
            if i == 20:
                break

        """
        Note: validate is not ready now!

        """
    end_str = ' '.join([str(m.mean) for m in metric_list.values()] +
                       [metric_list['batch_time'].total])
    end_epoch_str = "END epoch:{:<3d}".format(epoch)

    logger.info("{:s} {:s} {:s}s".format(
            coloring(end_epoch_str, "RED"),
            coloring("TRAIN", "PURPLE"),
            coloring(end_str, "OKGREEN")))

    logger.info('[TRAIN] training finished')
