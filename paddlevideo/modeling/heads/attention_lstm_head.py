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
import math
import sys

import paddle
import paddle.nn.functional as F
from paddle.nn import AdaptiveAvgPool2D, Linear, Dropout

from .base import BaseHead
from ..registry import HEADS
from ..weight_init import weight_init_
from ...metrics.youtube8m import eval_util as youtube8m_metrics

for_align = False #True

@HEADS.register()
class AttentionLstmHead(BaseHead):
    """AttentionLstmHead.
    Args: TODO
    """

    def __init__(self, num_classes=3862, feature_num=2, feature_dims=[1024, 128], embedding_size=512, lstm_size=1024,
                 in_channels=2048,
                 loss_cfg=dict(name='CrossEntropyLoss')):
        super(AttentionLstmHead, self).__init__(num_classes, in_channels, loss_cfg)
        self.num_classes = num_classes
        self.feature_dims = feature_dims
        self.embedding_size = embedding_size  #512
        self.lstm_size = lstm_size #1024
        self.feature_num = len(self.feature_dims)
        for i in range(self.feature_num): #0:rgb, 1:audio
            fc_feature = paddle.nn.Linear(in_features=self.feature_dims[i], out_features=self.embedding_size)
            self.add_sublayer("fc_feature{}".format(i), fc_feature)

            bi_lstm = paddle.nn.LSTM(input_size=self.embedding_size , hidden_size=self.lstm_size, direction="bidirectional")
            self.add_sublayer("bi_lstm{}".format(i), bi_lstm)

            drop_rate = 0.5
            self.dropout = paddle.nn.Dropout(drop_rate)

            att_fc = paddle.nn.Linear(in_features=self.lstm_size*2, out_features=1)
            self.add_sublayer("att_fc{}".format(i), att_fc)
            self.softmax = paddle.nn.Softmax()

        self.fc_out1 = paddle.nn.Linear(in_features=self.lstm_size*4, out_features=8192)
        self.relu = paddle.nn.ReLU()
        self.fc_out2 = paddle.nn.Linear(in_features=8192, out_features=4096)
        self.fc_logit = paddle.nn.Linear(in_features=4096, out_features=self.num_classes)
        self.sigmoid = paddle.nn.Sigmoid()
    def init_weights(self):
        pass
    def forward(self, inputs):
        #处理变长特征
        #1. 将特征padding为相同长度, 组成tensor
        #2. 1的同时, 构造一个与1中的tensor具有相同尺寸的mask tensor,其值为0或1,mask取值0代表该位置的tensor是padding生成的元素,取值1代表真实元素
        #3. 在网络适当位置,利用mask tensor参与计算, 使得网络输出与padding填充的元素无关
        assert ( len(inputs) == self.feature_num), "Input tensor does not contain {} features".format(self.feature_num)
        att_outs = []
        for i in range(len(inputs)):
            ###组网1. fc. 将输入特征向量映射到512长度
            m = getattr(self, "fc_feature{}".format(i))
            #i: 0 ,inputs[i].shape: [2, 256, 1024] [2]
            #i: 1 ,inputs[i].shape: [2, 256, 128] [2]
            output_fc = m(inputs[i][0])
            output_fc = paddle.tanh(output_fc)

            ###组网2. bi_lstm, 输出对应静态图的lstm_concat_0
            m = getattr(self, "bi_lstm{}".format(i))
            #inputs: [batch_size, time_steps(max_len), hidden_size]
            lstm_out, _ = m(inputs=output_fc, sequence_length=inputs[i][1])#x_data_rgb_len, x_data_audio_len

            #i: 0/1 lstm_out.shape: [2, 512, 2048] lstm_out[0/1]
            #[batch_size, time_steps(max_len), num_directions * hidden_size]
            if for_align == False:
                lstm_dropout = self.dropout(lstm_out)
            else:
                lstm_dropout = lstm_out

            ###组网3. att_fc
            m = getattr(self, "att_fc{}".format(i))
            lstm_weight = m(lstm_dropout) #[2, 256, 1]

            ###softmax replace start, for it's relevant to sum in time step
            lstm_exp = paddle.exp(lstm_weight)#[2, 256, 1]
            lstm_mask = paddle.mean(inputs[i][2], axis=2)#[2, 256]

            lstm_exp_with_mask = paddle.multiply(x=lstm_exp, y=lstm_mask, axis=0)#[2, 256, 1]

            lstm_sum_with_mask = paddle.sum(lstm_exp_with_mask, axis=1) #[2, 1]

            exponent = -1
            lstm_denominator = paddle.pow(lstm_sum_with_mask, exponent) #[2, 1]

            lstm_softmax = paddle.multiply(x=lstm_exp, y=lstm_denominator, axis=0)#[2, 256, 1]
            lstm_weight = lstm_softmax
            ###softmax replace end

            lstm_scale = paddle.multiply(x=lstm_dropout, y=lstm_weight, axis=0) #[2, 256, 2048]

            ###sequence_pool's replace start, for it's relevant to sum in time step
            lstm_scale_with_mask = paddle.multiply(x=lstm_scale, y=lstm_mask, axis=0)#[2, 256, 2048]
            fea_lens = inputs[i][1]
            fea_len = int(fea_lens[0])
            lstm_pool = paddle.sum(lstm_scale_with_mask, axis=1)
            ###sequence_pool's replace end
            att_outs.append(lstm_pool)
            #i: 0/1 lstm_weight.shape: [2, 256, 1] ,lstm_scale.shape: [2, 256, 2048] lstm_pool.shape: [2, 2048] lstm.py 234
        att_out = paddle.concat(att_outs, axis=1) #[2, 4096]
        fc_out1 = self.fc_out1(att_out)# [2, 8192]
        fc_out1_act = self.relu(fc_out1)
        fc_out2 = self.fc_out2(fc_out1_act)
        fc_out2_act = paddle.tanh(fc_out2)
        fc_logit = self.fc_logit(fc_out2_act)
        output = self.sigmoid(fc_logit)
        #shapes:, att_out: [2, 4096] fc_out1: [2, 8192] fc_out2: [2, 4096] fc_logit: [2, 3862] output: [2, 3862] lstm.py 242
        return fc_logit, output


    def loss(self,
             lstm_logit,
             labels,
             reduce_sum=False,
             return_loss=True,
             **kwargs):
        labels.stop_gradient = True
        losses = dict()
        if return_loss:
            cost = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(x=lstm_logit, label=labels)
            cost = paddle.sum(cost, axis=-1)
            sum_cost = paddle.sum(cost)

        return sum_cost

    def metric(self,
             lstm_output,
             labels,
             reduce_sum=False,
             return_loss=True,
             **kwargs):
        pred = lstm_output.numpy()
        label = labels.numpy()
        hit_at_one = youtube8m_metrics.calculate_hit_at_one(pred, label)
        perr = youtube8m_metrics.calculate_precision_at_equal_recall_rate(pred, label)
        gap = youtube8m_metrics.calculate_gap(pred, label)
        return hit_at_one, perr, gap
