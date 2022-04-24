# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import random
from PIL import Image
from paddle.vision.transforms import CenterCrop
from ..registry import DATASETS
from .base import BaseDataset


@DATASETS.register()
class RESIDEDataset(BaseDataset):

    def __init__(self,
                 pipeline,
                 data_prefix="",
                 file_path="",
                 crop_size=240,
                 test_mode=False,
                 suffix='.png'):
        super().__init__(file_path, pipeline)
        self.data_prefix = data_prefix
        self.suffix = suffix
        self.test_mode = test_mode
        self.crop_size = crop_size

    def load_file(self):
        """load file, abstractmethod"""

        self.haze_imgs_dir = os.listdir(os.path.join(self.file_path, 'hazy'))
        self.haze_imgs = [
            os.path.join(self.file_path, 'hazy', img)
            for img in self.haze_imgs_dir
        ]
        self.clear_dir = os.path.join(self.file_path, 'clear')

    def prepare_train(self, idx):
        """Prepare the frames for training/valid given index. """
        path = self.haze_imgs[idx]
        haze = Image.open(path)
        if not isinstance(self.crop_size, str):
            while haze.size[0] < self.crop_size or haze.size[1] < self.crop_size:
                index = random.randint(0, len(self.haze_imgs))
                haze = Image.open(self.haze_imgs[index])
                path = self.haze_imgs[index]
        if sys.platform == 'win32':
            id = path.split('\\')[-1].split('_')[0]
        else:
            id = path.split('/')[-1].split('_')[0]
        clear_name = id + self.suffix
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        clear = CenterCrop(haze.size[::-1])(clear)
        results = {'haze': haze, 'clear': clear}
        results = self.pipeline(results)
        return results['haze'], results['clear']

    def prepare_test(self, idx):
        """Prepare the frames for test given index. """
        path = self.haze_imgs[idx]
        haze = Image.open(path)
        if self.test_mode == False:
            while haze.size[0] < self.crop_size or haze.size[1] < self.crop_size:
                index = random.randint(0, len(self.haze_imgs))
                haze = Image.open(self.haze_imgs[index])
                path = self.haze_imgs[index]
        if sys.platform == 'win32':
            id = path.split('\\')[-1].split('_')[0]
        else:
            id = path.split('/')[-1].split('_')[0]
        clear_name = id + self.suffix
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        clear = CenterCrop(haze.size[::-1])(clear)
        results = {'haze': haze, 'clear': clear}
        results = self.pipeline(results)
        return results['haze'], results['clear']

    def __len__(self):
        """get the size of the dataset."""
        return len(self.haze_imgs)
