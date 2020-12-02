# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from ...registry import RECOGNIZERS
from .base import BaseRecognizer
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


@RECOGNIZERS.register()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""
    def forward_train(self, imgs, labels, **kwargs):
        """Define how the model is going to train, from input to output.
        """
        feature = self.extract_feature(imgs)
        cls_score = self.head(feature)
        loss_metrics = self.head.loss(cls_score, labels, **kwargs)
        return loss_metrics

    def forward_valid(self, imgs):
        """Define how the model is going to valid, from input to output."""
        feature = self.extract_feature(imgs)
        cls_score = self.head(feature)
        return cls_score

    def train_step(self, data_batch, **kwargs):
        """Training step.
        """
        imgs = [data_batch[0], data_batch[1]]
        labels = data_batch[2]

        # call forward
        loss_metrics = self(imgs, labels, return_loss=True)
        return loss_metrics

    def val_step(self, data_batch, **kwargs):
        """Validating setp.
        """
        imgs = [data_batch[0], data_batch[1]]
        labels = data_batch[2]

        # call forward
        loss_metrics = self(imgs, labels, return_loss=True)
        return loss_metrics
