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

from ...registry import MULTIMODAL
from .base import BaseMultimodal
import paddle
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


@MULTIMODAL.register()
class ActBert(BaseMultimodal):
    """ActBert model framework."""
    def forward_net(self, text_ids, action_feat, image_feat, image_loc,
                    token_type_ids, text_mask, image_mask, action_mask):
        pred = self.backbone(text_ids, action_feat, image_feat, image_loc,
                             token_type_ids, text_mask, image_mask, action_mask)
        return pred

    def train_step(self, data_batch):
        """Define how the model is going to train, from input to output.
        """
        pred = self.forward_net(data_batch)
        loss_metrics = self.loss(pred)
        return loss_metrics

    def val_step(self, data_batch):
        #     imgs = data_batch[0]
        #     labels = data_batch[1:]
        #     cls_score = self.forward_net(imgs)
        #     loss_metrics = self.head.loss(cls_score, labels, valid_mode=True)
        #     return loss_metrics
        pass

    def test_step(self, data_batch):
        """Define how the model is going to test, from input to output."""
        text_ids, action_feat, image_feat, image_loc, token_type_ids, text_mask, image_mask, action_mask = data_batch[:
                                                                                                                      -1]
        action_feat = action_feat.squeeze(0)
        image_feat = image_feat.squeeze(0)
        image_loc = image_loc.squeeze(0)
        image_mask = image_mask.squeeze(0)
        action_mask = action_mask.squeeze(0)
        prediction_scores_t, prediction_scores_v, prediction_scores_a, seq_relationship_score = self.forward_net(text_ids, \
            action_feat, image_feat, image_loc, token_type_ids, text_mask, image_mask, action_mask)
        return prediction_scores_t, prediction_scores_v, prediction_scores_a, seq_relationship_score

    def infer_step(self, data_batch):
        pass
        # """Define how the model is going to test, from input to output."""
        # imgs = data_batch[0]
        # cls_score = self.forward_net(imgs)
        # return cls_score
