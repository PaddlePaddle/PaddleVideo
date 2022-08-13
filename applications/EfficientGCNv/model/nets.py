import paddle
from paddle import nn
from .attentions import Attention_Layer
from .layers import Spatial_Graph_Layer, Temporal_Basic_Layer
from utils import *
import numpy as np
import math
from .initializer import kaiming_normal_, ones_, zeros_, normal_

class EfficientGCN(nn.Layer):
    def __init__(self, data_shape, block_args, fusion_stage, stem_channel, **kwargs):
        super(EfficientGCN, self).__init__()

        num_input, num_channel, _, _, _ = data_shape

        # input branches
        self.input_branches = nn.LayerList([EfficientGCN_Blocks(
            init_channel = stem_channel,
            block_args = block_args[:fusion_stage],
            input_channel = num_channel,
            **kwargs
        ) for _ in range(num_input)])

        # main stream
        last_channel = stem_channel if fusion_stage == 0 else block_args[fusion_stage-1][0]
        self.main_stream = EfficientGCN_Blocks(
            init_channel = num_input * last_channel,
            block_args = block_args[fusion_stage:],
            **kwargs
        )

        # output
        last_channel = num_input * block_args[-1][0] if fusion_stage == len(block_args) else block_args[-1][0]
        self.classifier = EfficientGCN_Classifier(last_channel, **kwargs)

        # init parameters
        init_param(self.sublayers())
    def forward(self, x):

        N, I, C, T, V, M = x.shape
        x = x.transpose((1, 0, 5, 2, 3, 4))
        x = x.reshape([I, N*M, C, T, V])
        # input branches
        x = paddle.concat([branch(x[i]) for i, branch in enumerate(self.input_branches)], axis=1)
        # main stream
        x = self.main_stream(x)
        # output
        _, C, T, V = x.shape
        feature = x.reshape([N, M, C, T, V]).transpose((0, 2, 3, 4, 1))
        out = self.classifier(feature).reshape([N, -1])

        return out, feature

class EfficientGCN_Blocks(nn.Sequential):
    def __init__(self, init_channel, block_args, layer_type, kernel_size, input_channel=0, **kwargs):
        super(EfficientGCN_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size

        if input_channel > 0:  # if the blocks in the input branches
            self.add_sublayer('init_bn', nn.BatchNorm2D(input_channel))
            self.add_sublayer('stem_scn', Spatial_Graph_Layer(input_channel, init_channel, max_graph_distance, **kwargs))
            self.add_sublayer('stem_tcn', Temporal_Basic_Layer(init_channel, temporal_window_size, **kwargs))

        last_channel = init_channel
        temporal_layer = import_class(f'model.layers.Temporal_{layer_type}_Layer')

        for i, [channel, stride, depth] in enumerate(block_args):
            self.add_sublayer(f'block-{i}_scn', Spatial_Graph_Layer(last_channel, channel, max_graph_distance, **kwargs))
            for j in range(depth):
                s = stride if j == 0 else 1
                self.add_sublayer(f'block-{i}_tcn-{j}', temporal_layer(channel, temporal_window_size, stride=s, **kwargs))
            self.add_sublayer(f'block-{i}_att', Attention_Layer(channel, **kwargs))
            last_channel = channel

class EfficientGCN_Classifier(nn.Sequential):
    def __init__(self, curr_channel, num_class, drop_prob, **kwargs):
        super(EfficientGCN_Classifier, self).__init__()

        self.add_sublayer('gap', nn.AdaptiveAvgPool3D(1))
        self.add_sublayer('dropout', nn.Dropout(drop_prob))
        self.add_sublayer('fc', nn.Conv3D(curr_channel, num_class, kernel_size=1))

def init_param(layers):
    for l in layers:
        if isinstance(l, nn.Conv1D) or isinstance(l, nn.Conv2D):

            # v = np.random.normal(loc=0.,scale=np.sqrt(2./n),size=l.weight.shape).astype('float32')
            l.weight.set_value(kaiming_normal_(l.weight, mode='fan_out'))
            # nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='leaky_relu')
            if l.bias is not None:
                # l.bias.set_value(0)
                l.bias.set_value(zeros_(l.bias))

        elif isinstance(l, nn.BatchNorm1D) or isinstance(l, nn.BatchNorm2D) or isinstance(l, nn.BatchNorm3D):
            l.weight.set_value(ones_(l.weight))
            l.bias.set_value(zeros_(l.bias))
        elif isinstance(l, nn.Conv3D) or isinstance(l, nn.Linear):
            l.weight.set_value(normal_(l.weight))
            l.bias.set_value(zeros_(l.bias))
            # nn.init.normal_(l.weight, std=0.001)
            # if l.bias is not None:
                # nn.init.constant_(l.bias, 0)


