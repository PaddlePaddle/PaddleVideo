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

import paddle
from paddlevideo.utils import get_logger
from ..loader.builder import build_dataloader, build_dataset
from ..metrics import build_metric
from ..modeling.builder import build_model
from paddlevideo.utils import (build_record, get_logger,
                               load, log_batch, mkdir, save)
import time

logger = get_logger("paddlevideo")


@paddle.no_grad()
def evaluate_model(cfg, weights, parallel=True):
    # 1. Construct model.
    if cfg.MODEL.backbone.get('pretrained'):
        cfg.MODEL.backbone.pretrained = ''  # disable pretrain model init
    model = build_model(cfg.MODEL)
    if parallel:
        model = paddle.DataParallel(model)
        
    # load trained weights 
    state_dicts = load(weights)
    model.set_state_dict(state_dicts)

    # 2. Construct dataset and dataloader.
    batch_size = cfg.DATASET.get("valid_batch_size", 8)
    places = paddle.set_device('gpu')

    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    num_workers = cfg.DATASET.get('valid_num_workers', num_workers)

    valid_dataset = build_dataset((cfg.DATASET.valid, cfg.PIPELINE.valid))
    validate_dataloader_setting = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        places=places,
        drop_last=False,
        shuffle=cfg.DATASET.get(
            'shuffle_valid',
            False)  #NOTE: attention lstm need shuffle valid data.
    )
    valid_loader = build_dataloader(valid_dataset,
                                    **validate_dataloader_setting)
    # 模型评估
    model.eval()

    results = []
    record_list = build_record(cfg.MODEL)
    record_list.pop('lr')
    tic = time.time()
    if parallel:
        rank = dist.get_rank()
    #single_gpu_test and multi_gpu_test
    for i, data in enumerate(valid_loader):
        outputs = model(data, mode='valid')
        if cfg.MODEL.framework == "FastRCNN":
            results.extend(outputs)

        #log_record
        if cfg.MODEL.framework != "FastRCNN":
            for name, value in outputs.items():
                record_list[name].update(value, batch_size)

        record_list['batch_time'].update(time.time() - tic)
        tic = time.time()

        ips = "ips: {:.5f} instance/sec.".format(
                        batch_size / record_list["batch_time"].val)
        log_batch(record_list, i,0, 0, "val", ips)
        
    if cfg.MODEL.framework == "FastRCNN":
        if parallel:
            results = collect_results_cpu(results, len(valid_dataset))
        if not parallel or (parallel and rank==0):
            eval_res = valid_dataset.evaluate( results) 
            for name, value in eval_res.items():
                record_list[name].update(value, batch_size)

    ips = "avg_ips: {:.5f} instance/sec.".format(
        batch_size * record_list["batch_time"].count /
        record_list["batch_time"].sum)
    print(record_list, "val", ips)

    best_flag = False
    if cfg.MODEL.framework == "FastRCNN" and (not parallel or (parallel and rank==0)):
        best = record_list["mAP@0.5IOU"].val 
        print("mAP@0.5IOU:",best)
        return
        
    #best2, cfg.MODEL.framework != "FastRCNN":
    for top_flag in ['hit_at_one', 'top1']:
        if record_list.get(
                top_flag) and record_list[top_flag].avg > best:
            best = record_list[top_flag].avg
            best_flag = True
    
    print(best)
    print(best_flag)
    


