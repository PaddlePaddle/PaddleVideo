#!/usr/bin/env python
# coding=utf-8
"""
attention lstm add ernie model
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

import os
import paddle
import paddle.static as static

from .ernie import ErnieConfig, ErnieModel


class AttentionLstmErnie(object):
    """
    Base on scenario-classify (image + audio), add text information
    use ERNIE to extract text feature
    """
    def __init__(self, name, cfg, mode='train'):
        self.cfg = cfg
        self.name = name
        self.mode = mode
        self.py_reader = None
        self.get_config()

    def get_config(self):
        """get_config
        """
        # get model configs
        self.feature_num = self.cfg.MODEL.feature_num
        self.feature_names = self.cfg.MODEL.feature_names
        self.feature_dims = self.cfg.MODEL.feature_dims
        self.feature_dtypes = self.cfg.MODEL.feature_dtypes
        self.feature_lod_level = self.cfg.MODEL.feature_lod_level
        self.num_classes = self.cfg.MODEL.num_classes
        self.embedding_size = self.cfg.MODEL.embedding_size
        self.lstm_size_img = self.cfg.MODEL.lstm_size_img
        self.lstm_size_audio = self.cfg.MODEL.lstm_size_audio
        self.ernie_freeze = self.cfg.MODEL.ernie_freeze
        self.lstm_pool_mode = self.cfg.MODEL.lstm_pool_mode
        self.drop_rate = self.cfg.MODEL.drop_rate
        self.loss_type = self.cfg.TRAIN.loss_type
        self.ernie_pretrain_dict_path = self.cfg.TRAIN.ernie_pretrain_dict_path

        # get mode configs
        self.batch_size = self.get_config_from_sec(self.mode, 'batch_size', 1)
        self.num_gpus = self.get_config_from_sec(self.mode, 'num_gpus', 1)

        if self.mode == 'train':
            self.learning_rate = self.get_config_from_sec(
                'train', 'learning_rate', 1e-3)
            self.weight_decay = self.get_config_from_sec(
                'train', 'weight_decay', 8e-4)
            self.num_samples = self.get_config_from_sec(
                'train', 'num_samples', 5000000)
            self.decay_epochs = self.get_config_from_sec(
                'train', 'decay_epochs', [5])
            self.decay_gamma = self.get_config_from_sec(
                'train', 'decay_gamma', 0.1)

    def get_config_from_sec(self, sec, item, default=None):
        """get_config_from_sec"""
        if sec.upper() not in self.cfg:
            return default
        return self.cfg[sec.upper()].get(item, default)

    def build_input(self, use_pyreader):
        """
        build input
        """
        self.feature_input = []
        for name, dim, dtype, lod_level in zip(self.feature_names,
                                               self.feature_dims,
                                               self.feature_dtypes,
                                               self.feature_lod_level):
            self.feature_input.append(
                static.data(shape=dim,
                                  lod_level=lod_level,
                                  dtype=dtype,
                                  name=name))
        self.label_input = static.data(shape=[self.num_classes],
                                             dtype='float32',
                                             name='label')

        self.py_reader = paddle.fluid.io.PyReader(feed_list=self.feature_input +
                                           [self.label_input],
                                           capacity=1024,
                                           iterable=True)

    def ernie_encoder(self):
        """
        text feature extractor
        """
        ernie_config = ErnieConfig(
            os.path.join(self.ernie_pretrain_dict_path, 'ernie_config.json'))
        if self.mode != 'train':
            ernie_config['attention_probs_dropout_prob'] = 0.0
            ernie_config['hidden_dropout_prob'] = 0.0

        src_ids = self.feature_input[2][:, 0]
        sent_ids = self.feature_input[2][:, 1]
        position_ids = self.feature_input[2][:, 2]
        task_ids = self.feature_input[2][:, 3]
        input_mask = self.feature_input[2][:, 4].astype('float32')
        ernie = ErnieModel(src_ids=src_ids,
                           position_ids=position_ids,
                           sentence_ids=sent_ids,
                           task_ids=task_ids,
                           input_mask=input_mask,
                           config=ernie_config)
        enc_out = ernie.get_sequence_output()
        # to Freeze ERNIE param
        if self.ernie_freeze is True:
            enc_out.stop_gradient = True
        # ernie cnn
        enc_out_cnn = ernie.get_sequence_textcnn_output(enc_out, input_mask)

        enc_out_cnn_drop = paddle.nn.functional.dropout(enc_out_cnn, p=self.drop_rate, training=(self.mode=='train'))
        return enc_out_cnn_drop

    def build_model(self):
        """build_model
        """
        # ---------------- transfer from old paddle ---------------
        # get image,audio,text feature
        video_input_tensor = self.feature_input[0]
        audio_input_tensor = self.feature_input[1]
        self.ernie_feature = self.ernie_encoder()

        # ------image------
        lstm_forward_fc = static.nn.fc(x=video_input_tensor,
                                          size=self.lstm_size_img * 4,
                                          activation=None,
                                          bias_attr=False)
        lstm_forward, _ = paddle.fluid.layers.dynamic_lstm(input=lstm_forward_fc,
                                                    size=self.lstm_size_img *
                                                    4,
                                                    is_reverse=False,
                                                    use_peepholes=True)

        lsmt_backward_fc = static.nn.fc(x=video_input_tensor,
                                           size=self.lstm_size_img * 4,
                                           activation=None,
                                           bias_attr=None)
        lstm_backward, _ = paddle.fluid.layers.dynamic_lstm(input=lsmt_backward_fc,
                                                     size=self.lstm_size_img *
                                                     4,
                                                     is_reverse=True,
                                                     use_peepholes=True)

        lstm_forward_img = paddle.concat(
            x=[lstm_forward, lstm_backward], axis=1)

        lstm_dropout = paddle.nn.functional.dropout(lstm_forward_img, p=self.drop_rate, training=(self.mode=='train'))
        if self.lstm_pool_mode == 'text_guide':
            lstm_weight = self.attention_weight_by_feature_seq2seq_attention(
                self.ernie_feature, lstm_dropout, self.lstm_size_img * 2)
        else:
            lstm_weight = static.nn.fc(x=lstm_dropout,
                                          size=1,
                                          activation='sequence_softmax',
                                          bias_attr=None)
        scaled = paddle.multiply(x=lstm_dropout,
                                              y=lstm_weight)
        self.lstm_pool = paddle.static.nn.sequence_pool(input=scaled,
                                                    pool_type='sum')
        # ------audio------
        lstm_forward_fc_audio = static.nn.fc(
            x=audio_input_tensor,
            size=self.lstm_size_audio * 4,
            activation=None,
            bias_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(coeff=0.0),
                initializer=paddle.nn.initializer.Normal(std=0.0)))
        lstm_forward_audio, _ = paddle.fluid.layers.dynamic_lstm(
            input=lstm_forward_fc_audio,
            size=self.lstm_size_audio * 4,
            is_reverse=False,
            use_peepholes=True)

        lsmt_backward_fc_audio = static.nn.fc(x=audio_input_tensor,
                                                 size=self.lstm_size_audio * 4,
                                                 activation=None,
                                                 bias_attr=False)
        lstm_backward_audio, _ = paddle.fluid.layers.dynamic_lstm(
            input=lsmt_backward_fc_audio,
            size=self.lstm_size_audio * 4,
            is_reverse=True,
            use_peepholes=True)

        lstm_forward_audio = paddle.concat(
            x=[lstm_forward_audio, lstm_backward_audio], axis=1)

        lstm_dropout_audio = paddle.nn.functional.dropout(lstm_forward_audio, p=self.drop_rate, training=(self.mode=='train'))
        if self.lstm_pool_mode == 'text_guide':
            lstm_weight_audio = self.attention_weight_by_feature_seq2seq_attention(
                self.ernie_feature, lstm_dropout_audio,
                self.lstm_size_audio * 2)
        else:
            lstm_weight_audio = static.nn.fc(x=lstm_dropout_audio,
                                                size=1,
                                                activation='sequence_softmax',
                                                bias_attr=None)
        scaled_audio = paddle.multiply(x=lstm_dropout_audio,
                                                    y=lstm_weight_audio)
        self.lstm_pool_audio = paddle.static.nn.sequence_pool(input=scaled_audio,
                                                          pool_type='sum')

        lstm_concat = paddle.concat(
            x=[self.lstm_pool, self.lstm_pool_audio, self.ernie_feature],
            axis=1,
            name='final_concat')

        # lstm_concat = self.add_bn(lstm_concat)
        if self.loss_type == 'softmax':
            self.fc = static.nn.fc(x=lstm_concat,
                                      size=self.num_classes,
                                      activation='softmax')
        elif self.loss_type == 'sigmoid':
            self.fc = static.nn.fc(x=lstm_concat,
                                      size=self.num_classes,
                                      activation=None)
            self.logit = self.fc
            self.fc = paddle.nn.functional.sigmoid(self.fc)

        self.network_outputs = [self.fc]

    def attention_weight_by_feature_seq2seq_attention(
            self,
            text_feature,
            sequence_feature,
            sequence_feature_dim,
            name_prefix="seq2seq_attention"):
        """
        caculate weight by feature
        Neural Machine Translation by Jointly Learning to Align and Translate
        """
        text_feature_expand = paddle.static.nn.sequence_expand(text_feature,
                                                           sequence_feature,
                                                           ref_level=0)
        sequence_text_concat = paddle.concat(
            x=[sequence_feature, text_feature_expand],
            axis=-1,
            name='video_text_concat')
        energy = static.nn.fc(x=sequence_text_concat,
                                 size=sequence_feature_dim,
                                 activation='tanh',
                                 name=name_prefix + "_tanh_fc")
        weight_vector = static.nn.fc(x=energy,
                                        size=1,
                                        activation='sequence_softmax',
                                        bias_attr=None,
                                        name=name_prefix + "_softmax_fc")
        return weight_vector

    def add_bn(self, lstm_concat):
        """
        v2.5 add drop out and batch norm
        """
        input_fc_proj = static.nn.fc(
            x=lstm_concat,
            size=8192,
            activation=None,
            bias_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(coeff=0.0),
                initializer=paddle.nn.initializer.Normal(std=0.0)))
        input_fc_proj_bn = paddle.static.nn.batch_norm(
            input=input_fc_proj,
            act="relu",
            is_test=(not self.mode == 'train'))
        input_fc_proj_dropout = paddle.nn.functional.dropout(
            input_fc_proj_bn,
            p=self.drop_rate,
            training=(self.mode=='train'))
        input_fc_hidden = static.nn.fc(
            x=input_fc_proj_dropout,
            size=4096,
            activation=None,
            bias_attr=paddle.ParamAttr(
                regularizer=paddle.regularizer.L2Decay(coeff=0.0),
                initializer=paddle.nn.initializer.Normal(std=0.0)))
        input_fc_hidden_bn = paddle.static.nn.batch_norm(
            input=input_fc_hidden,
            act="relu",
            is_test=(not self.mode == 'train'))
        input_fc_hidden_dropout = paddle.nn.functional.dropout(
            input_fc_hidden_bn,
            p=self.drop_rate,
            training=(self.mode=='train'))
        return input_fc_hidden_dropout

    def optimizer(self):
        """
        optimizer
        """
        assert self.mode == 'train', "optimizer only can be get in train mode"
        values = [
            self.learning_rate * (self.decay_gamma ** i)
            for i in range(len(self.decay_epochs) + 1)
        ]
        iter_per_epoch = self.num_samples / self.batch_size
        boundaries = [e * iter_per_epoch for e in self.decay_epochs]
        return paddle.optimizer.RMSProp(
            learning_rate=paddle.optimizer.lr.PiecewiseDecay(values=values,
                                                       boundaries=boundaries),
            centered=True,
            weight_decay=paddle.regularizer.L2Decay(
                coeff=self.weight_decay))

    def softlabel_cross_entropy_loss(self):
        """
        softlabel_cross_entropy_loss
        """
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        '''
        cost = paddle.nn.functional.cross_entropy(input=self.network_outputs[0], \
                                          label=self.label_input)
        '''
        cost = paddle.nn.functional.cross_entropy(input=self.network_outputs[0], \
                                          label=self.label_input,
                                          soft_label=True)

        cost = paddle.sum(x=cost, axis=-1)
        sum_cost = paddle.sum(x=cost)
        self.loss_ = paddle.scale(sum_cost,
                                        scale=self.num_gpus,
                                        bias_after_scale=False)

        return self.loss_

    def sigmoid_cross_entropy_loss(self):
        """
        sigmoid_cross_entropy_loss
        """
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        cost = paddle.nn.functional.binary_cross_entropy(input=self.logit,\
                                          label=self.label_input, reduction=None)

        cost = paddle.sum(x=cost, axis=-1)
        sum_cost = paddle.sum(x=cost)
        self.loss_ = paddle.scale(sum_cost,
                                        scale=self.num_gpus,
                                        bias_after_scale=False)

        return self.loss_

    def loss(self):
        """
        loss
        """
        if self.loss_type == 'sigmoid':
            return self.sigmoid_cross_entropy_loss()
        else:
            return self.softlabel_cross_entropy_loss()

    def outputs(self):
        """
        get outputs
        """
        return self.network_outputs

    def feeds(self):
        """
        get feeds
        """
        return self.feature_input if self.mode == 'infer' else self.feature_input + [
            self.label_input
        ]

    def pyreader(self):
        """pyreader"""
        return self.py_reader

    def epoch_num(self):
        """get train epoch num"""
        return self.cfg.TRAIN.epoch

    def load_test_weights_file(self, exe, weights, prog, place):
        """
        load_test_weights_file
        """
        load_vars = [x for x in prog.list_vars() \
                     if isinstance(x, paddle.framework.Parameter)]
        static.load_vars(exe,
                           dirname=weights,
                           vars=load_vars,
                           filename="param")
