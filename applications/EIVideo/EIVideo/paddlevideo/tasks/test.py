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
from EIVideo.paddlevideo.utils import get_logger, load
from ..loader.builder import build_dataloader, build_dataset
from ..metrics import build_metric
from ..modeling.builder import build_model
from ..modeling.framework import Manet

logger = get_logger("paddlevideo")


@paddle.no_grad()
def test_model(cfg, weights, parallel=True):
    """Test model entry

    Args:
        cfg (dict): configuration.
        weights (str): weights path to load.
        parallel (bool): Whether to do multi-cards testing. Default: True.

    """
    if cfg.MODEL.framework == "Manet":
        cfg_helper = {"knns": 1, "is_save_image": True}
        cfg.update(cfg_helper)
        final = Manet().test_step(**cfg, weights=weights, parallel=False)
        return final
