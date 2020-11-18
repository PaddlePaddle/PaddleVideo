#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import cv2
import random
from PIL import Image

from ..registry import PIPELINES


@PIPELINES.register()
class VideoDecoder(object):
    """
    Decode mp4 file to frames.
    Args:
        filepath: the file path of mp4 file
    """

    def __init__(self):
        pass

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        #XXX get info from results!!!
        file_path = results['filename']
        cap = cv2.VideoCapture(file_path)
        videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampledFrames = []
        for i in range(videolen):
            ret, frame = cap.read()
            # maybe first frame is empty
            if ret == False:
                continue
            img = frame[:, :, ::-1]
            sampledFrames.append(img)
        results['frames'] = sampledFrames
        results['frames_len'] = videolen
        results['format'] = 'video'
        return results

@PIPELINES.register()
class FrameDecoder(object):
    """just parse results
    """
    def __init__(self):
        pass
    def __call__(self, results):
        results['format'] = 'frame'
        return results
