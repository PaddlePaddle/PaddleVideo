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
import sys

@DATASETS.register()
class FeatureDataset(BaseDataset):
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
    def __init__(self,
                 file_path,
                 pipeline,
                 data_prefix=None,
                 valid_mode=False,
                 suffix=None):
        self.suffix = suffix
        super().__init__(file_path,
                         pipeline,
                         data_prefix,
                         valid_mode)

    def load_file(self):
        """Load index file to get video information."""
        #print("chaj self.data_prefix:",self.data_prefix, "self.suffix:",self.suffix,__file__,sys._getframe().f_lineno)
        info = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                #line_split = line.strip().split()
                filename = line.strip() #line_split
                #print("chaj filename:",filename,__file__,sys._getframe().f_lineno)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                #print("chaj filename:",filename,__file__,sys._getframe().f_lineno)
                if self.suffix is not None:
                    filename = filename + self.suffix
                #print("chaj filename:",filename,__file__,sys._getframe().f_lineno)

                info.append(dict(filename=filename))
        return info
