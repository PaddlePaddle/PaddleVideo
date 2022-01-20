"""
Exclude from autoreload
%aimport -util.utils
"""
import os
import json
import random
from pathlib import Path
from datetime import datetime
from typing import List
from itertools import repeat
from collections import OrderedDict

import numpy as np
import paddle
import psutil
import humanize
from PIL import Image
from typeguard import typechecked


@typechecked
def filter_cmd_args(cmd_args: List[str], remove: List[str]) -> List[str]:
    drop = []
    for key in remove:
        if key not in cmd_args:
            continue
        pos = cmd_args.index(key)
        drop.append(pos)
        if len(cmd_args) > (pos + 1) and not cmd_args[pos + 1].startswith("--"):
            drop.append(pos + 1)
    for pos in reversed(drop):
        cmd_args.pop(pos)
    return cmd_args


@typechecked
def set_seeds(seed: int):
    """Set seeds for randomisation libraries.

    Args:
        seed: the seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def memory_summary():
    vmem = psutil.virtual_memory()
    msg = (
        f">>> Currently using {vmem.percent}% of system memory "
        f"{humanize.naturalsize(vmem.used)}/{humanize.naturalsize(vmem.available)}"
    )
    print(msg)


def flatten_dict(x, keysep="-"):
    flat_dict = {}
    for key, val in x.items():
        if isinstance(val, dict):
            flat_subdict = flatten_dict(val)
            flat_dict.update({f"{key}{keysep}{subkey}": subval
                              for subkey, subval in flat_subdict.items()})
        else:
            flat_dict.update({key: val})
    return flat_dict


def expert_tensor_storage(experts, feat_aggregation):
    expert_storage = {"fixed": set(), "variable": set(), "flaky": set()}
    # fixed_sz_experts, variable_sz_experts, flaky_experts = set(), set(), set()
    for expert, config in feat_aggregation.items():
        if config["temporal"] in {"vlad",  "fixed_seg"}:
            expert_storage["variable"].add(expert)
        elif config["temporal"] in {"avg", "max", "avg-max", "max-avg", "avg-max-ent", 
                                    "max-avg-ent"}:
            expert_storage["fixed"].add(expert)
        else:
            raise ValueError(f"unknown temporal strategy: {config['temporal']}")
        # some "flaky" experts are only available for a fraction of videos - we need
        # to pass this information (in the form of indices) into the network for any
        # experts present in the current dataset
        if config.get("flaky", False):
            expert_storage["flaky"].add(expert)

    # we only allocate storage for experts used by the current dataset
    for key, value in expert_storage.items():
        expert_storage[key] = value.intersection(set(experts))
    return expert_storage


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def path2str(x):
    """Recursively convert pathlib objects to strings to enable serialization"""
    for key, val in x.items():
        if isinstance(val, dict):
            path2str(val)
        elif isinstance(val, Path):
            x[key] = str(val)


def write_json(content, fname, paths2strs=False):
    if paths2strs:
        path2str(content)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class HashableDict(dict):
    def __hash__(self):
        return hash(frozenset(self))


class HashableOrderedDict(dict):
    def __hash__(self):
        return hash(frozenset(self))


def compute_trn_config(config, logger=None):
    trn_config = {}
    feat_agg = config["data_loader"]["args"]["feat_aggregation"]
    for static_expert in feat_agg.keys():
        if static_expert in feat_agg:
            if "trn_seg" in feat_agg[static_expert].keys():
                trn_config[static_expert] = feat_agg[static_expert]["trn_seg"]
    return trn_config


def compute_dims(config, logger=None):
    if logger is None:
        logger = config.get_logger('utils')

    experts = config["experts"]
    # TODO(Samuel): clean up the logic since it's a little convoluted
    ordered = sorted(config["experts"]["modalities"])

    if experts["drop_feats"]:
        to_drop = experts["drop_feats"].split(",")
        logger.info(f"dropping: {to_drop}")
        ordered = [x for x in ordered if x not in to_drop]

    feat_agg = config["data_loader"]["args"]["feat_aggregation"]
    dims = []
    arch_args = config["arch"]["args"]
    vlad_clusters = arch_args["vlad_clusters"]
    for expert in ordered:
        temporal = feat_agg[expert]["temporal"]
        if expert == "face":
            in_dim, out_dim = experts["face_dim"], experts["face_dim"]
        elif expert == "features_scene" and temporal == "vlad":
            in_dim, out_dim = 2208 * vlad_clusters["features_scene"], 2208
        elif expert == "features_s3d" and temporal == "vlad":
            in_dim, out_dim = 1024 * vlad_clusters["features_s3d"], 1024
        elif expert == "features_flow" and temporal == "vlad":
            in_dim, out_dim = 1024 * vlad_clusters["features_flow"], 1024
        elif expert == "features_rgb" and temporal == "vlad":
            in_dim, out_dim = 2048 * vlad_clusters["features_rgb"], 2048
        elif expert == "features_ocr" and temporal == "vlad":
            in_dim, out_dim = 300 * vlad_clusters["features_ocr"], 300
        elif expert == "features_face" and temporal == "vlad":
            in_dim, out_dim = 512 * vlad_clusters["features_face"], 512
        elif expert == "features_speech" and temporal == "vlad":
            in_dim, out_dim = 300 * vlad_clusters["features_speech"], 300
        elif expert == "features_audio" and temporal == "vlad":
            in_dim, out_dim = 128 * vlad_clusters["features_audio"], 128
        elif expert == "audio" and temporal == "vlad":
            in_dim, out_dim = 128 * vlad_clusters["audio"], 128
        elif expert == "audio" and temporal == "vlad":
            in_dim, out_dim = 128 * vlad_clusters["audio"], 128
        elif expert == "speech" and temporal == "vlad":
            in_dim, out_dim = 300 * vlad_clusters["speech"], 300
        elif expert == "ocr" and temporal == "vlad":
            in_dim, out_dim = 300 * vlad_clusters["ocr"], 300
        elif expert == "detection":
            # allow for avg pooling
            det_clusters = arch_args["vlad_clusters"].get("detection", 1)
            in_dim, out_dim = 1541 * det_clusters, 1541
        elif expert == "detection-sem":
            if config["data_loader"]["args"].get("spatial_feats", False):
                base = 300 + 16
            else:
                base = 300 + 5
            det_clusters = arch_args["vlad_clusters"].get("detection-sem", 1)
            in_dim, out_dim = base * det_clusters, base
        elif expert == "openpose":
            base = 54
            det_clusters = arch_args["vlad_clusters"].get("openpose", 1)
            in_dim, out_dim = base * det_clusters, base
        else:
            common_dim = feat_agg[expert]["feat_dims"][feat_agg[expert]["type"]]
            # account for aggregation of multilpe forms (e.g. avg + max pooling)
            common_dim = common_dim * len(feat_agg[expert]["temporal"].split("-"))
            in_dim, out_dim = common_dim, common_dim

        # For the CE architecture, we need to project all features to a common
        # dimensionality
        if arch_args.get("mimic_ce_dims", False):
            out_dim = experts["ce_shared_dim"]

        dims.append((expert, (in_dim, out_dim)))
    expert_dims = OrderedDict(dims)

    if vlad_clusters["text"] == 0:
        msg = "vlad can only be disabled for text with single tokens"
        assert config["data_loader"]["args"]["max_tokens"]["text"] == 1, msg

    if config["experts"]["text_agg"] == "avg":
        msg = "averaging can only be performed with text using single tokens"
        assert config["arch"]["args"]["vlad_clusters"]["text"] == 0
        assert config["data_loader"]["args"]["max_tokens"]["text"] == 1

    # To remove the dependency of dataloader on the model architecture, we create a
    # second copy of the expert dimensions which accounts for the number of vlad
    # clusters
    raw_input_dims = OrderedDict()
    for expert, dim_pair in expert_dims.items():
        raw_dim = dim_pair[0]
        if expert in {"audio", "speech", "ocr", "detection", "detection-sem", "openpose", "features_audio", "features_speech", "features_face", "features_ocr",  "features_rgb", "features_flow", "features_s3d", "features_scene",
                      "speech.mozilla.0"}:
            if feat_agg[expert]["temporal"] == "vlad":
                raw_dim = raw_dim // vlad_clusters.get(expert, 1)
        raw_input_dims[expert] = raw_dim

    return expert_dims, raw_input_dims


def ensure_tensor(x):
    if not isinstance(x, paddle.Tensor): #if not isinstance(x, torch.Tensor):
        x = paddle.to_tensor(x) #    x = torch.from_numpy(x)
    return x


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, paddle.Tensor): #if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image #image_tensor = input_image.data
        else:
            return input_image
        # convert it into a numpy array
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # post-processing: tranpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
