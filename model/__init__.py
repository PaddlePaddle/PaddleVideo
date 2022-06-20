import math
from paddle import nn

from . import layers
from .nets import EfficientGCN
from .activations import *


__activations = {
    'relu': nn.ReLU(),
    'relu6': nn.ReLU6(),
    'hswish': HardSwish(inplace=True),
    'swish': Swish(inplace=True),
}

def rescale_block(block_args, scale_args, scale_factor):
    channel_scaler = math.pow(scale_args[0], scale_factor)
    depth_scaler = math.pow(scale_args[1], scale_factor)
    new_block_args = []
    for [channel, stride, depth] in block_args:
        channel = max(int(round(channel * channel_scaler / 16)) * 16, 16)
        depth = int(round(depth * depth_scaler))
        new_block_args.append([channel, stride, depth])
    return new_block_args

def create(model_type, act_type, block_args, scale_args, **kwargs):
    kwargs.update({
        'act': __activations[act_type],
        'block_args': rescale_block(block_args, scale_args, int(model_type[-1])),
    })
    return EfficientGCN(**kwargs)