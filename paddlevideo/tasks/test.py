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

logger = get_logger("paddlevideo")
def test_model(model,
               dataset,
               cfg,
               weight,
               parallel=False):
    """Test model entry

    Args:
        model (paddle.nn.Layer): The model to be tested.
        dataset (paddle.dataset): Train dataaset.


    """
    batch_size = cfg.DATASET.get("batch_size", 2)
    places = paddle.set_device('gpu')

    dataloader_setting = dict(
        batch_size = batch_size,
        num_worker = cfg.DATASET.get("num_worker", 0)
        places = places,
        drop_last = False,
        shuffle = False)

    data_loader = build_dataloader(dataset, **dataloader_setting)

    model.eval()

    state_dicts = paddle.load(weight)
    model.set_state_dict(state_dicts)

    if parallel:
        model = paddle.DataParallel(model)
    top1 = []
    for data in data_loader:
        with paddle.fluid.dygraph.no_grad():
            if parallel:
                outputs = model._layers.test_step(data)
            else:
                outputs = model.test_step(data)
        top1.append(outputs('top1'))
        logger.info(outputs)
    logger.info(np.mean(top1))

        
