import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../../")))

from paddlevideo.loader.pipelines import (CenterCrop, Image2Array,
                                          Normalization, Sampler, Scale,
                                          VideoDecoder, TenCrop)
import numpy as np
from typing import Dict, Tuple, List, Callable

VALID_MODELS = ["PPTSM", "PPTSN"]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """softmax function

    Args:
        x (np.ndarray): logits
        axis (int): axis

    Returns:
        np.ndarray: probs
    """
    x -= np.max(x, axis=axis, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    return x


def preprocess_PPTSM(video_path: str) -> Tuple[Dict[str, np.ndarray], List]:
    """preprocess

    Args:
        video_path (str): input video path

    Returns:
        Tuple[Dict[str, np.ndarray], List]: feed and fetch
    """
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    seq = Compose([
        VideoDecoder(),
        Sampler(8, 1, valid_mode=True),
        Scale(256),
        CenterCrop(224),
        Image2Array(),
        Normalization(img_mean, img_std)
    ])
    results = {"filename": video_path}
    results = seq(results)
    tmp_inp = np.expand_dims(results["imgs"], axis=0)  # [b,t,c,h,w]
    tmp_inp = np.expand_dims(tmp_inp, axis=0)  # [1,b,t,c,h,w]
    feed = {"data_batch_0": tmp_inp}
    fetch = ["outputs"]
    return feed, fetch


def preprocess_PPTSN(video_path: str) -> Tuple[Dict[str, np.ndarray], List]:
    """preprocess

    Args:
        video_path (str): input video path

    Returns:
        Tuple[Dict[str, np.ndarray], List]: feed and fetch
    """
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    seq = Compose([
        VideoDecoder(),
        Sampler(25, 1, valid_mode=True, select_left=True),
        Scale(256, fixed_ratio=True, do_round=True, backend='cv2'),
        TenCrop(224),
        Image2Array(),
        Normalization(img_mean, img_std)
    ])
    results = {"filename": video_path}
    results = seq(results)
    tmp_inp = np.expand_dims(results["imgs"], axis=0)  # [b,t,c,h,w]
    tmp_inp = np.expand_dims(tmp_inp, axis=0)  # [1,b,t,c,h,w]
    feed = {"data_batch_0": tmp_inp}
    fetch = ["outputs"]
    return feed, fetch


def get_preprocess_func(model_name: str) -> Callable:
    """get preprocess function by model_name

    Args:
        model_name (str): model's name, must in `VALID_MODELS`


    Returns:
        Callable: preprocess function corresponding to model name
    """
    if model_name == "PPTSM":
        return preprocess_PPTSM
    elif model_name == "PPTSN":
        return preprocess_PPTSN
    else:
        raise ValueError(
            f"model_name must in {VALID_MODELS}, but got model_name")
