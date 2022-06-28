"""
ernie reader
"""
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import sys
import os
import json
import random
import logging
import numpy as np
import six
from io import open
from collections import namedtuple

from .tokenization import FullTokenizer, convert_to_unicode

log = logging.getLogger(__name__)

if six.PY3:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def csv_reader(fd, delimiter='\t'):
    """csv_reader
    """
    def gen():
        """gen
        """
        for i in fd:
            yield i.rstrip('\n').split(delimiter)

    return gen()


class BaseReader(object):
    """BaseReader
    """
    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 is_inference=False,
                 random_seed=None,
                 tokenizer="FullTokenizer",
                 is_classify=True,
                 is_regression=False,
                 for_cn=True,
                 task_id=0):
        self.max_seq_len = max_seq_len
        self.tokenizer = FullTokenizer(vocab_file=vocab_path,
                                       do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens
        self.is_inference = is_inference
        self.for_cn = for_cn
        self.task_id = task_id

        np.random.seed(random_seed)

        self.is_classify = is_classify
        self.is_regression = is_regression
        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        if label_map_config:
            with open(label_map_config, encoding='utf8') as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`."""

        text_a = convert_to_unicode(example.text_a)
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None

        has_text_b = False
        if isinstance(example, dict):
            has_text_b = "text_b" in example.keys()
        else:
            has_text_b = "text_b" in example._fields

        if has_text_b:
            text_b = convert_to_unicode(example.text_b)
            tokens_b = tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT/ERNIE is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        text_type_ids = []
        tokens.append("[CLS]")
        text_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            text_type_ids.append(0)
        tokens.append("[SEP]")
        text_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                text_type_ids.append(1)
            tokens.append("[SEP]")
            text_type_ids.append(1)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))

        if self.is_inference:
            Record = namedtuple('Record',
                                ['token_ids', 'text_type_ids', 'position_ids'])
            record = Record(token_ids=token_ids,
                            text_type_ids=text_type_ids,
                            position_ids=position_ids)
        else:
            if self.label_map:
                label_id = self.label_map[example.label]
            else:
                label_id = example.label

            Record = namedtuple('Record', [
                'token_ids', 'text_type_ids', 'position_ids', 'label_id', 'qid'
            ])

            qid = None
            if "qid" in example._fields:
                qid = example.qid

            record = Record(token_ids=token_ids,
                            text_type_ids=text_type_ids,
                            position_ids=position_ids,
                            label_id=label_id,
                            qid=qid)
        return record

    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            record = self._convert_example_to_record(example, self.max_seq_len,
                                                     self.tokenizer)
            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records)
                batch_records, max_len = [record], len(record.token_ids)

        if batch_records:
            yield self._pad_batch_records(batch_records)


class ExtractEmbeddingReader(BaseReader):
    """
    data prepare for getting erine embedding 
    """
    def _pad_batch_records(self, batch_records):
        """
        对字标号，位置标号特征进行固定长度补全
        batch_records 包含多条文本的标号
        return [字标号列表，文本类型列表，位置特征列表，任务标号列表，掩码列表]
        """
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [
            record.text_type_ids for record in batch_records
        ]
        batch_position_ids = [record.position_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True,
            max_len=self.max_seq_len)
        padded_text_type_ids = pad_batch_data(batch_text_type_ids,
                                              pad_idx=self.pad_id,
                                              max_len=self.max_seq_len)
        padded_position_ids = pad_batch_data(batch_position_ids,
                                             pad_idx=self.pad_id,
                                             max_len=self.max_seq_len)
        padded_task_ids = np.ones_like(padded_token_ids,
                                       dtype="int64") * self.task_id

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            padded_task_ids, input_mask
        ]
        return return_list

    def data_generate_from_text(self, text):
        """
        trans text to idx
        input single text
        return 5*maxlen*1
        """
        Example = namedtuple('Example', ['text_a', 'label'])
        example = Example(text, 0)
        records = [
            self._convert_example_to_record(example, self.max_seq_len,
                                            self.tokenizer)
        ]
        pad_records = self._pad_batch_records(records)
        text_one_hot = np.concatenate(pad_records, axis=0).astype('int64')
        return text_one_hot


def pad_batch_data(insts,
                   pad_idx=0,
                   max_len=None,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    if max_len is None:
        max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array(
            [[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape([-1])]

    return return_list if len(return_list) > 1 else return_list[0]
