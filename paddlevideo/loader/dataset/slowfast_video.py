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

from ..registry import DATASETS
from .base import BaseDataset
import os.path as osp

import copy
import numpy as np
import random


@DATASETS.register()
class SlowfastVideoDataset(BaseDataset):
    """Video dataset for action recognition
       The dataset loads raw videos and apply specified transforms on them.

       The index file is a file with multiple lines, and each line indicates
       a sample video with the filepath and label, which are split with a whitesapce.
       Example of a inde file:

       .. code-block:: txt

           path/000.mp4 1
           path/001.mp4 1
           path/002.mp4 2
           path/003.mp4 2

       Args:
           file_path(str): Path to the index file.
           pipeline(XXX): A sequence of data transforms.
           **kwargs: Keyword arguments for ```BaseDataset```.

    """
    def __init__(
        self,
        file_path,
        pipeline,
        #                 data_prefix=None,
        #                 valid_mode=False,
        num_ensemble_views=1,
        num_spatial_crops=1,
        num_retries=5,
        **kwargs,
    ):
        super().__init__(file_path, pipeline, **kwargs)
        #        self.file_path = file_path
        #        self.data_prefix = osp.realpath(data_prefix) if \
        #            osp.isdir(data_prefix) else data_prefix
        #        self.valid_mode = valid_mode

        self.num_ensemble_views = num_ensemble_views  #diff in infer
        self.num_spatial_crops = num_spatial_crops  #diff in infer
        self._num_clips = (self.num_ensemble_views * self.num_spatial_crops)

        self.num_retries = num_retries
        #        self.pipeline = pipeline
        self.info = self.load_file()

    def load_file(self):
        """Load index file to get video information."""
        info = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                filename, labels = line_split
                #if self.data_prefix is not None:
                #    filename = osp.join(self.data_prefix, filename)
                for tidx in range(self.num_ensemble_views):
                    for sidx in range(self.num_spatial_crops):
                        info.append(
                            dict(
                                filename=filename,
                                labels=int(labels),
                                temporal_sample_index=tidx if self.valid_mode
                                else -1,  # to move to yaml file
                                spatial_sample_index=sidx
                                if self.valid_mode else -1,
                                temporal_num_clips=self.num_ensemble_views,
                                spatial_num_clips=self.num_spatial_crops,
                            ))
        return info

    def prepare_train(
        self,
        idx,
    ):
        """Prepare the frames for training given the index."""
        for ir in range(self.num_retries):
            try:
                results = copy.deepcopy(self.info[idx])
                #Note: For now, paddle.io.DataLoader cannot support dict type retval, so convert to list here
                result = self.pipeline(results)  #hj: try except??
            except Exception as e:
                print(e)  # hj: log error
                if ir < self.num_retries - 1:
                    print("Error when loading {}, have {} trys, will try again".
                          format(results['filename'], ir))
                idx = random.randint(0, len(self.info) - 1)
                continue
        #XXX have to unsqueeze label here or before calc metric!
        return results['imgs'][0], results['imgs'][1], np.array(
            [results['labels']])  #hj: when to unpack

    def prepare_valid(self, idx):
        """Prepare the frames for valid given the index."""
        #        results = copy.deepcopy(self.info[idx])
        #        #Note: For now, paddle.io.DataLoader cannot support dict type retval, so convert to list here
        #        results = self.pipeline(results)
        for ir in range(self.num_retries):
            try:
                results = copy.deepcopy(self.info[idx])
                #Note: For now, paddle.io.DataLoader cannot support dict type retval, so convert to list here
                result = self.pipeline(results)  #hj: try except??
            except Exception as e:
                print(e)  # hj: log error
                if ir < self.num_retries - 1:
                    print("Error when loading {}, have {} trys, will try again".
                          format(results['filename'], ir))
                idx = random.randint(0, len(self.info) - 1)
                continue
        return results['imgs'][0], results['imgs'][1], np.array(
            [results['labels']])
