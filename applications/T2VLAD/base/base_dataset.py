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

import time
import json
import random
import paddle
import inspect
import logging
import functools
import data_loader

import numpy as np
import pickle as pkl

from pathlib import Path
from abc import abstractmethod
from typing import Dict, Union
from numpy.random import randint
from typeguard import typechecked
from collections import OrderedDict
from zsvision.zs_utils import memcache
try:
    from paddlenlp.transformers import BertTokenizer
except ImportError as e:
    print(
        f"{e}, [paddlenlp] package and it's dependencies is required for T2VLAD."
    )
from utils import ensure_tensor, expert_tensor_storage

# For SLURM usage, buffering makes it difficult to see events as they happen, so we set
# the global print statement to enforce flushing
print = functools.partial(print, flush=True)


class BaseDataset(paddle.io.Dataset):
    @staticmethod
    @abstractmethod
    @typechecked
    def dataset_paths() -> Dict[str, Union[Path, str]]:
        """Generates a datastructure containing all the paths required to load features
        """
        raise NotImplementedError

    @abstractmethod
    def sanity_checks(self):
        """Run sanity checks on loaded data
        """
        raise NotImplementedError

    @abstractmethod
    def load_features(self):
        """Load features from disk
        """
        raise NotImplementedError

    @typechecked
    def __init__(
        self,
        data_dir: Path,
        eval_only: bool,
        use_zeros_for_missing: bool,
        text_agg: str,
        text_feat: str,
        split_name: str,
        cls_partition: str,
        root_feat_folder: str,
        text_dim: int,
        num_test_captions: int,
        restrict_train_captions: int,
        max_tokens: Dict[str, int],
        logger: logging.Logger,
        raw_input_dims: Dict[str, int],
        feat_aggregation: Dict[str, Dict],
    ):
        self.eval_only = eval_only
        self.logger = logger
        self.text_feat = text_feat
        self.data_dir = data_dir
        self.text_dim = text_dim
        self.restrict_train_captions = restrict_train_captions
        self.max_tokens = max_tokens
        self.cls_partition = cls_partition
        self.num_test_captions = num_test_captions
        self.feat_aggregation = feat_aggregation
        self.root_feat = data_dir / root_feat_folder
        self.experts = set(raw_input_dims.keys())
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # This attributes can be overloaded by different datasets, so it must be set
        # before the `load_features() method call`
        self.restrict_test_captions = None
        self.text_features = None
        self.label_features = None
        self.video_labels = None
        self.raw_captions = None
        self.features = None

        self.word2int = json.load(open('word2int.json'))

        # Use a single caption per video when forming training minibatches (different
        # captions from the same video may still be used across different minibatches)
        self.captions_per_video = 1

        self.ordered_experts = list(raw_input_dims.keys())

        # Training and test lists are set by dataset-specific subclasses
        self.partition_lists = {}
        self.configure_train_test_splits(split_name=split_name)

        # All retrieval-based tasks use a single dataloader (and handle the retrieval
        # data separately), whereas for classification we use one dataloader for
        # training and one for validation.
        self.logger.info("The current task is retrieval")
        self.sample_list = self.partition_lists["train"]
        self.num_samples = len(self.sample_list)
        num_val = len(self.partition_lists["val"])

        self.raw_input_dims = raw_input_dims

        # we store default paths to enable visualisations (this can be overloaded by
        # dataset-specific classes)
        self.video_path_retrieval = [
            f"videos/{x}.mp4" for x in self.partition_lists["val"]
        ]

        # NOTE: We use nans rather than zeros to indicate missing faces, unless we wish
        # to test single modality strength, which requires passing zeroed features for
        # missing videos
        if use_zeros_for_missing:
            self.MISSING_VAL = 0
        else:
            self.MISSING_VAL = np.nan

        # load the dataset-specific features into memory
        self.load_features()

        if text_agg == "avg":
            self.logger.info("averaging the text features...")
            for key, val in self.text_features.items():
                self.text_features[key] = [
                    np.mean(x, 0, keepdims=1) for x in val
                ]
            self.logger.info("finished averaging the text features")

        self.trn_config = {}
        self.raw_config = {}
        self.tensor_storage = expert_tensor_storage(self.experts,
                                                    self.feat_aggregation)
        for static_expert in self.tensor_storage["fixed"]:
            if static_expert in self.feat_aggregation:
                if "trn_seg" in self.feat_aggregation[static_expert].keys():
                    self.trn_config[static_expert] = \
                        self.feat_aggregation[static_expert]["trn_seg"]
                if "raw" in self.feat_aggregation[static_expert]["temporal"]:
                    self.raw_config[static_expert] = 1

        retrieval = {
            expert: np.zeros(
                (num_val, self.max_tokens[expert], raw_input_dims[expert]))
            for expert in self.tensor_storage["variable"]
        }
        retrieval.update({
            expert: np.zeros((num_val, raw_input_dims[expert]))
            for expert in self.tensor_storage["fixed"]
        })
        self.retrieval = retrieval
        self.test_ind = {
            expert: paddle.ones([num_val])
            for expert in self.experts
        }
        self.raw_captions_retrieval = [None] * num_val

        # avoid evaluation on missing queries
        self.query_masks = np.zeros((num_val, num_test_captions))
        self.text_token_mask = np.zeros((num_val, num_test_captions))
        self.text_retrieval = np.zeros((num_val, self.num_test_captions,
                                        self.max_tokens["text"], self.text_dim))
        self.cap_retrieval = paddle.zeros(
            [num_val, self.num_test_captions, self.max_tokens["text"]],
            dtype='int64'
        )  #self.cap_retrieval = th.zeros((num_val, self.num_test_captions, self.max_tokens["text"]))
        self.att_retrieval = paddle.zeros(
            [num_val, self.num_test_captions, self.max_tokens["text"]],
            dtype='int64'
        )  #self.att_retrieval = th.zeros((num_val, self.num_test_captions, self.max_tokens["text"]))

        save_cap = []
        for ii, video_name in enumerate(self.partition_lists["val"]):

            self.raw_captions_retrieval[ii] = self.raw_captions[video_name]
            for expert in self.tensor_storage["fixed"].intersection(
                    self.experts):
                feats = self.features[expert][video_name]
                drop = self.has_missing_values(feats)
                self.test_ind[expert][ii] = not drop
                self.retrieval[expert][ii] = feats
                if drop:
                    self.retrieval[expert][ii][:] = self.MISSING_VAL
                if self.feat_aggregation[expert].get("binarise", False):
                    keep = np.logical_not(
                        np.isnan(self.retrieval[expert][:, 0, 0]))
                    marker = np.ones_like(self.retrieval[expert][keep])
                    self.retrieval[expert][keep] = marker

            for expert in self.tensor_storage["variable"].intersection(
                    self.experts):
                feats = self.features[expert][video_name]
                drop = self.has_missing_values(feats)
                self.test_ind[expert][ii] = not drop
                if drop:
                    self.retrieval[expert][ii][:] = self.MISSING_VAL
                if self.feat_aggregation[expert].get("binarise", False):
                    keep = np.logical_not(
                        np.isnan(self.retrieval[expert][:, 0, 0]))
                    marker = np.ones_like(self.retrieval[expert][keep])
                    self.retrieval[expert][keep] = marker
                if self.test_ind[expert][ii]:
                    keep = min(self.max_tokens[expert], len(feats))
                    self.retrieval[expert][ii, :keep, :] = feats[:keep]

            candidates_sentences = self.text_features[video_name]
            if self.restrict_test_captions is not None:
                keep_sent_idx = self.restrict_test_captions[video_name]
                candidates_sentences = [candidates_sentences[keep_sent_idx]]

            self.query_masks[ii, :len(candidates_sentences)] = 1

            for test_caption_idx in range(self.num_test_captions):
                if len(candidates_sentences) <= test_caption_idx:
                    break
                keep = min(len(candidates_sentences[test_caption_idx]),
                           self.max_tokens["text"])
                self.text_token_mask[ii, test_caption_idx] = keep
                sent = self.raw_captions_retrieval[ii][test_caption_idx]
                sent = " ".join(sent)
                sent = sent.strip()
                encoded_dict = self.tokenizer.__call__(
                    sent,
                    max_seq_len=self.max_tokens["text"],
                    pad_to_max_seq_len=True,
                    return_attention_mask=True,
                    truncation_strategy='longest_first')
                cap_ids = paddle.to_tensor(encoded_dict['input_ids'])
                attention_mask = paddle.to_tensor(
                    encoded_dict['attention_mask'])
                save_cap.append(sent)
                self.cap_retrieval[ii, test_caption_idx, :] = cap_ids
                self.att_retrieval[ii, test_caption_idx, :] = attention_mask
                if ii % 500 == 0 and test_caption_idx == 0:
                    msg = (
                        f"{ii}/{len(self.partition_lists['val'])} will evaluate "
                        f"sentence {test_caption_idx} out of "
                        f"{len(candidates_sentences)} (has {keep} words) "
                        f"{video_name}")
                    self.logger.info(msg)
                text_feats = candidates_sentences[test_caption_idx][:keep]
                if text_feats.shape[0] == 0:
                    text_feats = 0
                    raise ValueError("empty text features!")
                self.text_retrieval[ii, test_caption_idx, :keep, :] = text_feats
        with open('run_cap.pkl', 'wb') as f:
            pkl.dump(save_cap, f)
        self.sanity_checks()

    def configure_train_test_splits(self, split_name):
        """Partition the datset into train/val/test splits.

        Args:
            split_name (str): the name of the split
        """
        self.paths = type(self).dataset_paths()
        print("loading training/val splits....")
        tic = time.time()
        for subset, path in self.paths["subset_list_paths"][split_name].items():
            root_feat = Path(self.root_feat)
            subset_list_path = root_feat / path
            if subset == "train" and self.eval_only:
                rows = []
            else:
                with open(subset_list_path) as f:
                    rows = f.read().splitlines()
            self.partition_lists[subset] = rows
        print("done in {:.3f}s".format(time.time() - tic))
        self.split_name = split_name

    def collate_data(self, data):
        batch_size = len(data)
        tensors = {}
        for expert in self.tensor_storage["fixed"]:
            if expert in self.trn_config.keys():
                tensors[expert] = paddle.to_tensor(
                    np.zeros((batch_size, self.trn_config[expert],
                              self.raw_input_dims[expert])))
            else:
                tensors[expert] = paddle.to_tensor(
                    np.zeros((batch_size, self.raw_input_dims[expert])))

        # Track which indices of each modality are available in the present batch
        ind = {
            expert: paddle.to_tensor(np.zeros(batch_size))
            for expert in self.experts
        }
        tensors.update({
            expert: paddle.to_tensor(
                np.zeros((batch_size, self.max_tokens[expert],
                          self.raw_input_dims[expert])))
            for expert in self.tensor_storage["variable"]
        })

        text_tensor = paddle.to_tensor(
            np.zeros((batch_size, self.captions_per_video,
                      self.max_tokens["text"], self.text_dim)))
        text_token_mask = paddle.to_tensor(
            np.zeros((batch_size, self.captions_per_video)))
        text_cap_id = paddle.zeros([batch_size, self.max_tokens["text"]],
                                   dtype='int64')
        text_att_mask = paddle.zeros([batch_size, self.max_tokens["text"]],
                                     dtype='int64')

        for ii, _ in enumerate(data):
            datum = data[ii]
            for expert in self.experts:
                ind[expert][ii] = datum[f"{expert}_ind"]
            for expert in self.tensor_storage["fixed"]:
                tensors[expert][ii] = datum[expert]
            for expert in self.tensor_storage["variable"]:
                if ind[expert][ii]:
                    keep = min(len(datum[expert]), self.max_tokens[expert])
                    if keep:
                        tensors[expert][ii, :keep, :] = datum[expert][:keep]
                else:
                    tensors[expert][ii, :, :] = self.MISSING_VAL

            text = datum["text"]
            cap_id = datum["cap_id"]
            att_mask = datum["att_mask"]
            text_cap_id[ii, :] = paddle.to_tensor(cap_id)
            text_att_mask[ii, :] = paddle.to_tensor(att_mask)
            for jj in range(self.captions_per_video):
                keep = min(len(text[jj]), self.max_tokens["text"])
                text_tensor[ii, jj, :keep, :] = text[jj][:keep]
                text_token_mask[ii, jj] = keep

        ind = {key: ensure_tensor(val) for key, val in ind.items()}
        experts = OrderedDict(
            (expert, paddle.to_tensor(tensors[expert], dtype='float32'))
            for expert in self.ordered_experts)

        for expert in self.experts:
            if self.feat_aggregation[expert].get("binarise", False):
                replace = np.logical_not(paddle.isnan(experts[expert][:, 0, 0]))
                experts[expert][replace] = paddle.ones_like(
                    experts[expert][replace])

        minibatch = {"experts": experts, "ind": ind}
        minibatch["text"] = paddle.to_tensor(text_tensor, dtype='float32')
        minibatch["cap_id"] = paddle.to_tensor(text_cap_id, dtype='int64')
        minibatch["att_mask"] = paddle.to_tensor(text_att_mask, dtype='int64')
        minibatch["text_token_mask"] = paddle.to_tensor(text_token_mask)
        return minibatch

    def process_sent(self, sent, max_words, EOS: int = 1, UNK: int = 2):
        # set EOS=1, UNK=2 by default, consistent with file 'word2int.json'.
        tokens = [self.word2int.get(w, UNK) for w in sent]
        tokens = tokens[:max_words]
        tokens_len = len(tokens)
        tokens = np.array(tokens + [EOS] * (max_words - tokens_len))
        return tokens, tokens_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < self.num_samples:
            vid = self.sample_list[idx]
            features = {}
            for expert in self.experts:
                if expert not in self.trn_config.keys():
                    if expert in self.raw_config.keys():
                        features[expert] = np.mean(self.features[expert][vid],
                                                   axis=0)
                    else:
                        features[expert] = self.features[expert][vid]
                else:
                    raw_frame_feats = self.features[expert][vid]
                    new_length = 1
                    num_frames = raw_frame_feats.shape[0]
                    avg_duration = ((num_frames - new_length + 1) //
                                    self.trn_config[expert])
                    assert avg_duration > 0, "average duration must be positive"
                    if avg_duration > 0:
                        # maybe we could change to use average for each tiny segment
                        # seems like use everything per iter
                        offsets = np.multiply(
                            list(range(self.trn_config[expert])), avg_duration)
                        offsets += randint(avg_duration,
                                           size=self.trn_config[expert])
                        new_frame_feats = np.zeros(
                            (self.trn_config[expert], raw_frame_feats.shape[1]))
                        for idx, xx in enumerate(offsets):
                            new_frame_feats[idx, :] = raw_frame_feats[xx, :]
                        msg = "returning a wrong feature != segment num"
                        assert new_frame_feats.shape[0] == self.trn_config[
                            expert], msg
                        features[expert] = new_frame_feats

            ind = {}
            for expert in self.ordered_experts:
                if expert in self.tensor_storage["flaky"]:
                    ind[expert] = not self.has_missing_values(features[expert])
                else:
                    ind[expert] = 1

            # Handle some inconsistencies between how the text features are stored
            text = self.text_features[vid]
            if isinstance(text, list):
                pick = np.random.choice(len(text), size=self.captions_per_video)
                sent = self.raw_captions[vid][pick[0]]
                sent = " ".join(sent)
                sent = sent.strip()

                text = np.array(text)[pick]
                encoded_dict = self.tokenizer.__call__(
                    sent,
                    max_seq_len=self.max_tokens["text"],
                    pad_to_max_seq_len=True,
                    return_attention_mask=True,
                    truncation_strategy='longest_first')
                cap_id = encoded_dict['input_ids']
                token_type_ids = encoded_dict['token_type_ids']
                attention_mask = encoded_dict['attention_mask']
            else:
                pick = None
                text = np.random.choice(text, size=self.captions_per_video)

        # Return both the missing indices as well as the tensors
        sample = {"text": text}
        sample.update({"cap_id": cap_id})
        sample.update({"att_mask": attention_mask})
        sample.update({f"{key}_ind": val for key, val in ind.items()})
        sample.update(features)
        return sample

    def get_retrieval_data(self):
        experts = OrderedDict(
            (expert, paddle.to_tensor(self.retrieval[expert], dtype='float32'))
            for expert in self.ordered_experts)
        retrieval_data = {
            "text":
            paddle.to_tensor(ensure_tensor(self.text_retrieval),
                             dtype='float32'),
            "experts":
            experts,
            "cap_id":
            paddle.to_tensor(self.cap_retrieval, dtype='int64'),
            "att_mask":
            paddle.to_tensor(self.att_retrieval, dtype='int64'),
            "ind":
            self.test_ind,
            "text_token_mask":
            paddle.to_tensor(self.text_token_mask)
        }
        meta = {
            "query_masks": self.query_masks,
            "raw_captions": self.raw_captions_retrieval,
            "paths": self.video_path_retrieval,
        }
        return retrieval_data, meta

    def has_missing_values(self, x):
        return isinstance(x, float) and np.isnan(x)

    def visual_feat_paths(self, model_spec, tag=None):
        """Canonical path lookup for visual features
        """
        if model_spec not in self.ordered_experts:
            self.logger.info(
                f"Skipping load for {model_spec} (feature not requested)")
            return f"SKIPPED-{model_spec}"

        feat_type, model_name, _ = model_spec.split(".")
        aggs = self.feat_aggregation[model_spec]
        base = f"aggregated_{feat_type.replace('-', '_')}"
        required = ("fps", "pixel_dim", "stride")
        fps, pixel_dim, stride = [aggs.get(x, None) for x in required]
        if feat_type in {"facecrops", "faceboxes"}:
            base = f"{base}_{fps}fps_{pixel_dim}px_stride{stride}"
        elif feat_type not in {"ocr", "speech", "audio"}:
            base = f"{base}_{fps}fps_{pixel_dim}px_stride{stride}"

        for option in "offset", "inner_stride":
            if aggs.get(option, None) is not None:
                base += f"_{option}{aggs[option]}"

        feat_paths = []
        for agg in aggs["temporal"].split("-"):
            fname = f"{model_name}-{agg}"
            if aggs["type"] == "logits":
                fname = f"{fname}-logits"
            if tag is not None:
                fname += f"-{tag}"
            feat_paths.append(Path(base) / f"{fname}.pickle")
        return feat_paths

    def log_assert(self, bool_, msg="", verbose=True):
        """Use assertions that will be written to the logs. This is a recipe from:
        http://code.activestate.com/recipes/577074-logging-asserts/
        """
        try:
            assert bool_, msg
        except AssertionError:
            # construct an exception message from the code of the calling frame
            last_stackframe = inspect.stack()[-2]
            source_file, line_no, func = last_stackframe[1:4]
            source = f"Traceback (most recent call last):\n" + \
                     f" File {source_file}, line {line_no}, in {func}\n"
            if verbose:
                # include more lines than that where the statement was made
                source_code = open(source_file).readlines()
                source += "".join(source_code[line_no - 3:line_no + 1])
            else:
                source += last_stackframe[-2][0].strip()
            self.logger.debug(f"{msg}\n{source}")
            raise AssertionError(f"{msg}\n{source}")

    def summary_stats(self):
        """Report basic statistics about feature availability and variable lengths
        across the different subsets of the data.
        """
        self.logger.info("Computing feature stats...")
        queries = self.ordered_experts + ["text"]
        for subset, keep in self.partition_lists.items():
            keep = set(keep)
            print(f"Summary for {subset}")
            for expert in queries:
                if expert in self.features:
                    feats = self.features[expert]
                else:
                    feats = self.text_features
                vals = [feats[key] for key in keep]
                missing = 0
                sizes = []
                for val in vals:
                    if self.has_missing_values(val):
                        missing += 1
                    else:
                        sizes.append(len(val))
                if sizes:
                    stat_str = (f"min: {np.min(sizes):4}, "
                                f"max: {np.max(sizes):4}, "
                                f"mean: {np.mean(sizes):.1f}")
                    print(
                        f"{subset}: missing: {missing:4}, {stat_str} {expert}")
