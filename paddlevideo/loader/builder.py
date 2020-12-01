# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from paddle.io import DataLoader, DistributedBatchSampler
from .registry import DATASETS, PIPELINES
from ..utils.build_utils import build
from .pipelines.compose import Compose
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


def build_pipeline(cfg):
    """Build pipeline.
    Args:
        cfg (dict): root config dict.
    """
    return Compose(cfg)


def build_dataset(cfg):
    """Build dataset.
    Args:
        cfg (dict): root config dict.

    Returns:
        dataset: dataset.
    """
    #XXX: ugly code here!
    cfg_dataset, cfg_pipeline = cfg
    cfg_dataset.pipeline = build_pipeline(cfg_pipeline)
    dataset = build(cfg_dataset, DATASETS)
    return dataset


def build_dataloader(dataset,
                     batch_size,
                     num_workers,
                     places,
                     shuffle=True,
                     drop_last=True,
                     **kwargs):
    """Build Paddle Dataloader.

    XXX explain how the dataloader work!

    Args:
        dataset (paddle.dataset): A PaddlePaddle dataset object.
        batch_size (int): batch size on single card.
        num_worker (int): num_worker
        shuffle(bool): whether to shuffle the data at every epoch.
    """
    sampler = DistributedBatchSampler(dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      drop_last=drop_last)

    data_loader = DataLoader(dataset,
                             batch_sampler=sampler,
                             places=places,
                             num_workers=num_workers,
                             return_list=True,
                             **kwargs)

    return data_loader
