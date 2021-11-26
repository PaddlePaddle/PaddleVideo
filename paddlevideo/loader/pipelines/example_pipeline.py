#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from ..registry import PIPELINES


@PIPELINES.register()
class ExamplePipeline(object):
    """Example Pipeline """
    def __init__(self, mean=0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        data = results['data']
        label = results['label']
        norm_data = (data - self.mean) / self.std
        results['data'] = norm_data.astype('float32')
        results['label'] = label.astype('int64')
        return results
