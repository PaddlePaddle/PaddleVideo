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
from paddlevideo.utils import get_logger, load, build_record,log_batch

from ..loader.builder import build_dataloader, build_dataset
from ..metrics import build_metric
from ..modeling.builder import build_model
import time

logger = get_logger("paddlevideo")


@paddle.no_grad()
def test_model(cfg, weights, parallel=True):
    """Test model entry

    Args:
        cfg (dict): configuration.
        weights (str): weights path to load.
        parallel (bool): Whether to do multi-cards testing. Default: True.

    """
    # 1. Construct model.
    if cfg.MODEL.get('backbone') and cfg.MODEL.backbone.get('pretrained'):
        cfg.MODEL.backbone.pretrained = ''  # disable pretrain model init
    model = build_model(cfg.MODEL)

    if parallel:
        model = paddle.DataParallel(model)

    # 2. Construct dataset and dataloader.
    cfg.DATASET.test.test_mode = True
    dataset = build_dataset((cfg.DATASET.test, cfg.PIPELINE.test))
    batch_size = cfg.DATASET.get("test_batch_size", 8)
    places = paddle.set_device('gpu')
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    dataloader_setting = dict(batch_size=batch_size,
                              num_workers=num_workers,
                              places=places,
                              drop_last=False,
                              shuffle=False)

    data_loader = build_dataloader(dataset, **dataloader_setting)

    model.eval()

    state_dicts = load(weights)
    model.set_state_dict(state_dicts)

    if cfg.MODEL.framework != "FastRCNN":
        # add params to metrics
        cfg.METRIC.data_size = len(dataset)
        cfg.METRIC.batch_size = batch_size
        Metric = build_metric(cfg.METRIC)
    else:
        Metric = None
        
    results = []
    record_list = build_record(cfg.MODEL)
    record_list.pop('lr')
    tic = time.time()

    for batch_id, data in enumerate(data_loader):
        outputs = model(data, mode='test')
        if cfg.MODEL.framework == "FastRCNN":
            results.extend(outputs)
            record_list['batch_time'].update(time.time() - tic)
            tic = time.time()
            ips = "ips: {:.5f} instance/sec.".format(
                        batch_size / record_list["batch_time"].val)
            log_batch(record_list, batch_id,0, 0, "test", ips)
        else:
            Metric.update(batch_id, data, outputs)
    
    if cfg.MODEL.framework == "FastRCNN":
        if parallel:
            results = collect_results_cpu(results, len(dataset))
        if not parallel or (parallel and rank==0):
            test_res = dataset.evaluate( results) 
            for name, value in test_res.items():
                record_list[name].update(value, batch_size)
        
        best = record_list["mAP@0.5IOU"].val
        print(best)
    else:
        Metric.accumulate()
