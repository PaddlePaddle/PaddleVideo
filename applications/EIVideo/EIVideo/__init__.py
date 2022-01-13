# Author: Acer Zhang
# Datetime: 2022/1/6 
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import os
from EIVideo.version import __version__

EI_VIDEO_ROOT = os.path.abspath(os.path.dirname(__file__))
TEMP_IMG_SAVE_PATH = "./temp.png"
TEMP_JSON_SAVE_PATH = "./save.json"
TEMP_JSON_FINAL_PATH = "./final.json"


def join_root_path(path: str):
    return os.path.join(EI_VIDEO_ROOT, path)
