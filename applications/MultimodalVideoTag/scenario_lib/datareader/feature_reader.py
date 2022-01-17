"""
feature reader
"""
#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO
import numpy as np
import random
import os
import traceback
import pickle
python_ver = sys.version_info
from collections import defaultdict

import pandas as pd

from .ernie_task_reader import ExtractEmbeddingReader
from .reader_utils import DataReader


class FeatureReader(DataReader):
    """
    Data reader for youtube-8M dataset, which was stored as features extracted by prior networks
    This is for the three models: lstm, attention cluster, nextvlad

    dataset cfg: num_classes
                 batch_size
                 list
                 NextVlad only: eigen_file
    """
    def __init__(self, name, mode, cfg):
        """
        init
        """
        self.name = name
        self.mode = mode
        self.num_classes = cfg.MODEL.num_classes

        # set batch size and file list
        self.batch_size = cfg[mode.upper()]['batch_size']
        self.filelist = cfg[mode.upper()]['filelist']
        self.eigen_file = cfg.MODEL.get('eigen_file', None)
        self.num_seg = cfg.MODEL.get('num_seg', None)
        self.loss_type = cfg.TRAIN['loss_type']
        vocab_file = os.path.join(cfg.TRAIN.ernie_pretrain_dict_path,
                                  'vocab.txt')
        self.ernie_reader = ExtractEmbeddingReader(
            vocab_path=vocab_file,
            max_seq_len=cfg.MODEL.text_max_len,
            do_lower_case=True)
        url_title_label_file = cfg[mode.upper()]['url_title_label_file']
        self.class_dict = load_class_file(cfg.MODEL.class_name_file)
        self.url_title_info = load_video_file(url_title_label_file,
                                              self.class_dict, mode)

    def create_reader(self):
        """
        create reader
        """
        url_list = list(self.url_title_info.keys())
        if self.mode == 'train':
            random.shuffle(url_list)

        def reader():
            """reader
            """
            batch_out = []
            for url in url_list:
                try:
                    filepath = os.path.join(
                        self.filelist,
                        url.split('/')[-1].split('.')[0] + '.pkl')
                    if os.path.exists(filepath) is False:
                        continue
                    if python_ver < (3, 0):
                        record = pickle.load(open(filepath, 'rb'))
                    else:
                        record = pickle.load(open(filepath, 'rb'),
                                             encoding='iso-8859-1')
                    text_raw = self.url_title_info[url]['title']
                    rgb = record['feature']['image_pkl'].astype(float)
                    if record['feature']['audio_pkl'].shape[0] == 0:
                        audio_pkl = np.zeros((10, 128))
                        audio = audio_pkl.astype(float)
                    else:
                        audio = record['feature']['audio_pkl'].astype(float)
                    text_one_hot = self.ernie_reader.data_generate_from_text(
                        str(text_raw))
                    video = record['video']
                    if self.mode != 'infer':
                        label = self.url_title_info[url]['label']
                        label = [int(w) for w in label]
                        if self.loss_type == 'sigmoid':
                            label = make_one_hot(label, self.num_classes)
                        elif self.loss_type == 'softmax':
                            label = make_one_soft_hot(label, self.num_classes,
                                                      False)
                        batch_out.append((rgb, audio, text_one_hot, label))
                    else:
                        batch_out.append((rgb, audio, text_one_hot, video))
                    if len(batch_out) == self.batch_size:
                        yield batch_out
                        batch_out = []
                except Exception as e:
                    print("warning: load data {} failed, {}".format(
                        filepath, str(e)))
                    traceback.print_exc()
                    continue


# if self.mode == 'infer' and len(batch_out) > 0:
            if len(batch_out) > 0:
                yield batch_out

        return reader

    def get_config_from_sec(self, sec, item, default=None):
        """get_config_from_sec
        """
        if sec.upper() not in self.cfg:
            return default
        return self.cfg[sec.upper()].get(item, default)


def load_video_file(label_file, class_dict, mode='train'):
    """
    labelfile formate: URL \t title \t label1,label2
    return dict
    """
    data = pd.read_csv(label_file, sep='\t', header=None)
    url_info_dict = defaultdict(dict)
    for index, row in data.iterrows():
        url = row[0]
        if url in url_info_dict:
            continue
        if pd.isna(row[1]):
            title = ""
        else:
            title = str(row[1])
        if mode == 'infer':
            url_info_dict[url] = {'title': title}
        else:
            if pd.isna(row[2]):
                continue
            labels = row[2].split(',')
            labels_idx = [class_dict[w] for w in labels if w in class_dict]
            if len(labels_idx) < 1:
                continue
            if url not in url_info_dict:
                url_info_dict[url] = {'label': labels_idx, 'title': title}
    print('load video %d' % (len(url_info_dict)))
    return url_info_dict


def dequantize(feat_vector, max_quantized_value=2., min_quantized_value=-2.):
    """
    Dequantize the feature from the byte format to the float format
    """

    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value

    return feat_vector * scalar + bias


epsilon = 0.1
smmoth_score = (1.0 / float(210)) * epsilon


def label_smmoth(label_one_hot_vector):
    """
    label_smmoth
    """
    global smmoth_score
    for i in range(len(label_one_hot_vector)):
        if label_one_hot_vector[i] == 0:
            label_one_hot_vector[i] = smmoth_score
    return label_one_hot_vector


def make_one_soft_hot(label, dim=15, label_smmoth=False):
    """
    make_one_soft_hot
    """
    one_hot_soft_label = np.zeros(dim)
    one_hot_soft_label = one_hot_soft_label.astype(float)
    # multi-labelis
    # label smmoth
    if label_smmoth:
        one_hot_soft_label = label_smmoth(one_hot_soft_label)
    label_len = len(label)
    prob = (1 - np.sum(one_hot_soft_label)) / float(label_len)
    for ind in label:
        one_hot_soft_label[ind] += prob
    #one_hot_soft_label = label_smmoth(one_hot_soft_label)
    return one_hot_soft_label


def make_one_hot(label, dim=15):
    """
    make_one_hot
    """
    one_hot_soft_label = np.zeros(dim)
    one_hot_soft_label = one_hot_soft_label.astype(float)
    for ind in label:
        one_hot_soft_label[ind] = 1
    return one_hot_soft_label


def generate_random_idx(feature_len, num_seg):
    """
    generate_random_idx
    """
    idxs = []
    stride = float(feature_len) / num_seg
    for i in range(num_seg):
        pos = (i + np.random.random()) * stride
        idxs.append(min(feature_len - 1, int(pos)))
    return idxs


def get_batch_ernie_input_feature(reader, texts):
    """
    get_batch_ernie_input_feature
    """
    result_list = reader.data_generate_from_texts(texts)
    result_trans = []
    for i in range(len(texts)):
        result_trans.append([result_list[0][i],\
                             result_list[1][i],
                             result_list[2][i],
                             result_list[3][i],
                             result_list[4][i]])
    return np.array(result_trans)


def load_class_file(class_file):
    """
    load_class_file
    """
    class_lines = open(class_file, 'r', encoding='utf8').readlines()
    class_dict = {}
    for i, line in enumerate(class_lines):
        tmp = line.strip().split('\t')
        word = tmp[0]
        index = str(i)
        if len(tmp) == 2:
            index = tmp[1]
        class_dict[word] = index
    return class_dict


if __name__ == '__main__':
    pass
