# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import logging
import functools
from pathlib import Path
from typing import Dict, List

from typeguard import typechecked
from zsvision.zs_utils import memcache

from data_loader.MSRVTT_dataset import MSRVTT
from utils import HashableDict, HashableOrderedDict



@functools.lru_cache(maxsize=64, typed=False)
def dataset_loader(
        use_zeros_for_missing: bool,
        eval_only: bool,
        data_dir: str,
        text_agg: str,
        text_feat: str,
        split_name: str,
        dataset_name: str,
        cls_partition: str,
        root_feat_folder: str,
        text_dim: int,
        num_test_captions: int,
        restrict_train_captions: int,
        logger: logging.Logger,
        max_tokens: Dict[str, int],
        raw_input_dims: HashableOrderedDict,
        feat_aggregation: HashableDict,
):
    print(f"refreshing cache for {dataset_name} data loader [{split_name}]")
    kwargs = dict(
        data_dir=Path(data_dir),
        text_dim=text_dim,
        logger=logger,
        eval_only=eval_only,
        text_agg=text_agg,
        text_feat=text_feat,
        max_tokens=max_tokens,
        split_name=split_name,
        cls_partition=cls_partition,
        raw_input_dims=raw_input_dims,
        root_feat_folder=root_feat_folder,
        feat_aggregation=feat_aggregation,
        num_test_captions=num_test_captions,
        use_zeros_for_missing=use_zeros_for_missing,
        restrict_train_captions=restrict_train_captions,
    )
    if dataset_name == "MSRVTT":
        dataset = MSRVTT(**kwargs)
    return dataset


class ExpertDataLoader:

    @typechecked
    def __init__(
            self,
            eval_only: bool,
            use_zeros_for_missing: bool,
            text_dim: int,
            batch_size: int,
            num_workers: int,
            num_test_captions: int,
            data_dir: str,
            text_agg: str,
            text_feat: str,
            split_name: str,
            dataset_name: str,
            root_feat_folder: str,
            max_tokens: Dict[str, int],
            raw_input_dims: Dict[str, int],
            feat_aggregation: Dict[str, Dict],
            logger: logging.Logger,
            restrict_train_captions: int = 0,
            drop_last: bool = False,
            refresh_lru_cache: bool = False,
    ):

        # Ensure that the dictionaries are hashable to allow use of caching
        raw_input_dims = HashableOrderedDict(raw_input_dims)
        feat_aggregation = HashableDict(feat_aggregation)
        max_tokens = HashableDict(max_tokens)

        if refresh_lru_cache:
            logger.info("Explicitly refreshing dataloader and cuda cache")
            dataset_loader.cache_clear()
            memcache.cache_clear()

        common_kwargs = dict(
            logger=logger,
            data_dir=data_dir,
            text_dim=text_dim,
            text_agg=text_agg,
            eval_only=eval_only,
            text_feat=text_feat,
            max_tokens=max_tokens,
            dataset_name=dataset_name,
            split_name=split_name,
            root_feat_folder=root_feat_folder,
            use_zeros_for_missing=use_zeros_for_missing,
            num_test_captions=num_test_captions,
            raw_input_dims=raw_input_dims,
            feat_aggregation=feat_aggregation,
            restrict_train_captions=restrict_train_captions,
        )

        dataset = dataset_loader(cls_partition="train", **common_kwargs)
        x = dataset_loader.cache_info()  # pylint: disable=no-value-for-parameter
        logger.info(f"cache info {x}")
        self.dataloaders = {"dataset": dataset}
        self.dataloaders["retrieval"] = dataset.get_retrieval_data()
    
        if not eval_only:
            train_loader = paddle.io.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=dataset.collate_data,
                drop_last=drop_last,
                shuffle=True,
            )
            self.dataloaders["train"] = train_loader

        logger.info(f"Loading data loaders with {num_workers} workers")
        self.num_test_captions = num_test_captions
        self.dataset_name = dataset_name

    def __getitem__(self, key):
        return self.dataloaders[key]
