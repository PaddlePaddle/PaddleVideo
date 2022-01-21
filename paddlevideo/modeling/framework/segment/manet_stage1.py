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
from paddlevideo.loader.builder import build_pipeline
from paddlevideo.loader.pipelines import ToTensor_manet

import os
import timeit
import paddle
from PIL import Image
from davisinteractive.utils.scribbles import scribbles2mask, annotated_frames
from paddle import nn

from paddlevideo.utils import load
from paddlevideo.utils.manet_utils import float_, _palette, damage_masks, int_, long_, write_dict, rough_ROI, \
    load_video, get_images, submit_masks, get_scribbles
from ...builder import build_model
from ...registry import SEGMENT
from .base import BaseSegment


@SEGMENT.register()
class ManetSegment_Stage1(BaseSegment):
    def __init__(self, backbone=None, head=None, **cfg):
        super().__init__(backbone, head, **cfg)

    def train_step(self, data_batch, step, **cfg):
        """Define how the model is going to train, from input to output.
        返回任何你想打印到日志中的东西
        """
        ref_imgs = data_batch['ref_img']  # batch_size * 3 * h * w
        img1s = data_batch['img1']
        img2s = data_batch['img2']
        ref_scribble_labels = data_batch[
            'ref_scribble_label']  # batch_size * 1 * h * w
        label1s = data_batch['label1']
        label2s = data_batch['label2']
        seq_names = data_batch['meta']['seq_name']
        obj_nums = data_batch['meta']['obj_num']

        bs, _, h, w = img2s.shape
        inputs = paddle.concat((ref_imgs, img1s, img2s), 0)
        if self.train_cfg['damage_initial_previous_frame_mask']:
            try:
                label1s = damage_masks(label1s)
            except:
                label1s = label1s
                print('damage_error')

        tmp_dic = self.head(inputs,
                            ref_scribble_labels,
                            label1s,
                            use_local_map=True,
                            seq_names=seq_names,
                            gt_ids=obj_nums,
                            k_nearest_neighbors=self.train_cfg['knns'])
        label_and_obj_dic = {}
        label_dic = {}
        obj_dict = {}
        for i, seq_ in enumerate(seq_names):
            label_and_obj_dic[seq_] = (label2s[i], obj_nums[i])
        for seq_ in tmp_dic.keys():
            tmp_pred_logits = tmp_dic[seq_]
            tmp_pred_logits = nn.functional.interpolate(tmp_pred_logits,
                                                        size=(h, w),
                                                        mode='bilinear',
                                                        align_corners=True)
            tmp_dic[seq_] = tmp_pred_logits
            label_tmp, obj_num = label_and_obj_dic[seq_]
            label_dic[seq_] = long_(label_tmp)
        loss_metrics = {
            'loss':
            self.head.loss(dic_tmp=tmp_dic,
                           label_dic=label_dic,
                           step=step,
                           obj_dict=obj_dict) / bs
        }
        return loss_metrics

    def val_step(self, data_batch, **kwargs):
        pass

    def infer_step(self, data_batch, **kwargs):
        """Define how the model is going to test, from input to output."""
        pass

    def test_step(self, data_batch, **kwargs):
        pass
