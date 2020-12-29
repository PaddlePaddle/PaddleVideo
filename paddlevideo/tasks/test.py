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
from ..loader import build_dataloader
from ..metrics import build_metric

logger = get_logger("paddlevideo")


def test_model(model, dataset, cfg, weight, world_size):
    """Test model entry

    Args:
        model (paddle.nn.Layer): The model to be tested.
        dataset (paddle.dataset): Train dataaset.
    """
    #NOTE: add a new field : test_batch_size ?
    batch_size = cfg.DATASET.get("test_batch_size", 2)

    places = paddle.set_device('gpu')

    dataloader_setting = dict(batch_size=batch_size,
                              num_workers=cfg.DATASET.get("num_workers", 0),
                              places=places,
                              drop_last=False,
                              shuffle=False)

    data_loader = build_dataloader(dataset, **dataloader_setting)

    model.eval()

    state_dicts = paddle.load(weight)
    model.set_state_dict(state_dicts)

    parallel = world_size != 1
    if parallel:
        model = paddle.DataParallel(model)

    # add params to metrics
    cfg.METRIC.data_size = len(dataset)
    cfg.METRIC.batch_size = batch_size
    cfg.METRIC.world_size = world_size

    Metric = build_metric(cfg.METRIC)
    for batch_id, data in enumerate(data_loader):
        if parallel:
            outputs = model._layers.test_step(data)
        else:
            outputs = model.test_step(data)
        Metric.update(batch_id, data, outputs)
    Metric.accumulate()
