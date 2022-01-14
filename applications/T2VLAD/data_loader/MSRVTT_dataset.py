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

import copy

from pathlib import Path
from utils import memory_summary
from typeguard import typechecked
from typing import Dict, Union, List
from base.base_dataset import BaseDataset
from zsvision.zs_utils import memcache, concat_features

class MSRVTT(BaseDataset):
    @staticmethod
    @typechecked
    def dataset_paths() -> Dict[str, Union[str, List[str], Path, Dict]]:
        subset_paths = {}
        split_name = "jsfusion"
        train_list_path = "train_list_jsfusion.txt"
        test_list_path = "val_list_jsfusion.txt"
        # NOTE: The JSFusion split (referred to as 1k-A in the paper) uses all
        # videos, but randomly samples a single caption per video from the test
        # set for evaluation. To reproduce this evaluation, we use the indices
        # of the test captions, and restrict to this subset during eval.
        js_test_cap_idx_path = "jsfusion_val_caption_idx.pkl"
        subset_paths[split_name] = {"train": train_list_path, "val": test_list_path}
        custom_paths = {
            "features_audio": ["mmt_feats/features.audio.pkl"],
            "features_flow": ["mmt_feats/features.flow_agg.pkl"],
            "features_rgb": ["mmt_feats/features.rgb_agg.pkl"],
            "features_scene": ["mmt_feats/features.scene.pkl"],
            "features_face": ["mmt_feats/features.face_agg.pkl"],
            "features_ocr": ["mmt_feats/features.ocr.pkl"],
            "features_s3d": ["mmt_feats/features.s3d.pkl"],
            "features_speech": ["mmt_feats/features.speech.pkl"],
        }
        text_feat_paths = {
            "openai": "w2v_MSRVTT_openAIGPT.pickle",
        }
        text_feat_paths = {key: Path("aggregated_text_feats") / fname
                           for key, fname in text_feat_paths.items()}
        feature_info = {
            "custom_paths": custom_paths,
            "subset_list_paths": subset_paths,
            "text_feat_paths": text_feat_paths,
            "raw_captions_path": "raw-captions.pkl",
            "js_test_cap_idx_path": js_test_cap_idx_path,
        }
        return feature_info

    def load_features(self):
        root_feat = Path(self.root_feat)
        feat_names = {}
        custom_path_key = "custom_paths"
        feat_names.update(self.paths[custom_path_key])
        features = {}
        for expert, rel_names in feat_names.items():
            if expert not in self.ordered_experts:
                continue
            feat_paths = tuple([root_feat / rel_name for rel_name in rel_names])
            if len(feat_paths) == 1:
                features[expert] = memcache(feat_paths[0])
            else:
                # support multiple forms of feature (e.g. max and avg pooling). For
                # now, we only support direct concatenation
                msg = f"{expert}: Only direct concatenation of muliple feats is possible"
                print(f"Concatenating aggregates for {expert}....")
                is_concat = self.feat_aggregation[expert]["aggregate"] == "concat"
                self.log_assert(is_concat, msg=msg)
                axis = self.feat_aggregation[expert]["aggregate-axis"]
                x = concat_features.cache_info()  # pylint: disable=no-value-for-parameter
                print(f"concat cache info: {x}")
                features_ = concat_features(feat_paths, axis=axis)
                memory_summary()

                # Make separate feature copies for each split to allow in-place filtering
                features[expert] = copy.deepcopy(features_)

        self.features = features
        self.raw_captions = memcache(root_feat / self.paths["raw_captions_path"])
        text_feat_path = root_feat / self.paths["text_feat_paths"][self.text_feat]
        self.text_features = memcache(text_feat_path)

        if self.restrict_train_captions:
            # hash the video names to avoid O(n) lookups in long lists
            train_list = set(self.partition_lists["train"])
            for key, val in self.text_features.items():
                if key not in train_list:
                    continue

                if not self.split_name == "full-test":
                    # Note that we do not perform this sanity check for the full-test
                    # split, because the text features in the cached dataset will
                    # already have been cropped to the specified
                    # `resstrict_train_captions`
                    expect = {19, 20}
                    msg = f"expected train text feats as lists with length {expect}"
                    has_expected_feats = isinstance(val, list) and len(val) in expect
                    self.log_assert(has_expected_feats, msg=msg)

                # restrict to the first N captions (deterministic)
                self.text_features[key] = val[:self.restrict_train_captions]
        self.summary_stats()

    def sanity_checks(self):
        if self.num_test_captions == 20:
            if len(self.partition_lists["val"]) == 2990:
                missing = 6
            elif len(self.partition_lists["val"]) == 1000:
                missing = 2
            elif len(self.partition_lists["val"]) == 497:
                missing = 0
            else:
                raise ValueError("unrecognised test set")
            msg = "Expected to find two missing queries in MSRVTT for full eval"
            correct_missing = self.query_masks.sum() == self.query_masks.size - missing
            self.log_assert(correct_missing, msg=msg)
