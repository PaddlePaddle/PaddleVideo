# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Constant

from ...utils import load_ckpt
from ..registry import BACKBONES
from ..weight_init import trunc_normal_

__all__ = ['VisionTransformer']

zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def to_2tuple(x):
    return tuple([x] * 2)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    # issuecomment-532968956 ...
    See discussion: https://github.com/tensorflow/tpu/issues/494
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape).astype(x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor

    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        N, C = x.shape[1:]
        qkv = self.qkv(x).reshape(
            (-1, N, 3, self.num_heads, C // self.num_heads)).transpose(
                (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.1,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 num_segments = 8,
                 fold_div = 4):
                #attention_type='divided_space_time',
        super().__init__()
        self.n_seg = num_segments       #ckk
        self.foldP_div = fold_div       #ckk
        #self.attention_type = attention_type
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm1 = norm_layer(dim, epsilon=epsilon)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")

        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop)

        # Temporal Attention Parameters
        '''
        if self.attention_type == 'divided_space_time':
            if isinstance(norm_layer, str):
                self.temporal_norm1 = eval(norm_layer)(dim, epsilon=epsilon)
            elif isinstance(norm_layer, Callable):
                self.temporal_norm1 = norm_layer(dim, epsilon=epsilon)
            else:
                raise TypeError(
                    "The norm_layer must be str or paddle.nn.layer.Layer class")
            self.temporal_attn = Attention(dim,
                                           num_heads=num_heads,
                                           qkv_bias=qkv_bias,
                                           qk_scale=qk_scale,
                                           attn_drop=attn_drop,
                                           proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)
        '''
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm2 = norm_layer(dim, epsilon=epsilon)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
    # token_shift
    def shuift_tk(self, x):
        t = self.n_seg
        bt, n, c = x.shape
        b = bt // t
        x = x.reshape([b, t, n, c]) #B T N C
        
        fold = c // self.foldP_div
        out = paddle.zeros_like(x)
        out.stop_gradient = True
        # print("#### fold ", fold)
        # print(out.shape)
        # print(x[:, 1:, 0, :fold].unsqueeze(2).shape)
        # print(out[:, :-1, 0:1, :fold].shape)
        # exit(0)
        out[:, :-1, 0, :fold] = x[:, 1:, 0, :fold] # shift left
        out[:, 1:,  0, fold:2*fold] = x[:,:-1:, 0, fold:2*fold]
        
        out[:, :, 1:, :2*fold] = x[:, :, 1:, :2*fold]
        out[:, :, :, 2*fold:] = x[:, :, :, 2*fold:]
        
        return out.reshape([bt, n, c])
    
    def forward(self, x):
        x = self.shuift_tk(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.shuift_tk(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] //
                                                        patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(in_channels,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = x.transpose((0, 2, 1, 3, 4))
        x = x.reshape([-1, C, H, W])
        x = self.proj(x)
        W = x.shape[-1]
        x = x.flatten(2).transpose((0, 2, 1))
        return x, T, W


@BACKBONES.register()
class TokenShiftVisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch input
    """
    def __init__(self,
                 pretrained=None,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 num_seg=8,
                 attention_type='divided_space_time',
                 **args):

        super().__init__()
        self.pretrained = pretrained
        self.num_seg = num_seg
        self.attention_type = attention_type
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_channels=in_channels,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional Embeddings
        self.cls_token = self.create_parameter(shape=(1, 1, embed_dim),
                                               default_initializer=zeros_)
        self.pos_embed = self.create_parameter(shape=(1, num_patches + 1,
                                                      embed_dim),
                                               default_initializer=zeros_)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.attention_type != 'space_only':
            self.time_embed = self.create_parameter(shape=(1, num_seg,
                                                           embed_dim),
                                                    default_initializer=zeros_)
            self.time_drop = nn.Dropout(p=drop_rate)

        self.add_parameter("pos_embed", self.pos_embed)
        self.add_parameter("cls_token", self.cls_token)

        dpr = np.linspace(0, drop_path_rate, depth)

        self.blocks = nn.LayerList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  epsilon=epsilon,
                  num_segments= self.num_seg
                  ) for i in range(depth)
                #attention_type=self.attention_type
        ])

        self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)

    def init_weights(self):
        """First init model's weight"""
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_fn)

        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.sublayers(include_self=True):
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                        zeros_(m.temporal_fc.weight)
                        zeros_(m.temporal_fc.bias)
                    i += 1

        """Second, if provide pretrained ckpt, load it"""

        if isinstance(
                self.pretrained, str
        ) and self.pretrained.strip() != "":  # load pretrained weights
            load_ckpt(self,
                      self.pretrained,
                      num_patches=self.patch_embed.num_patches,
                      num_seg=self.num_seg,
                      attention_type=self.attention_type)

    def _init_fn(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            ones_(m.weight)
            zeros_(m.bias)

    def forward_features(self, x):
        # B = x.shape[0]
        B = paddle.shape(x)[0]
        x, T, W = self.patch_embed(x)  # [BT,nH*nW,F]
        cls_tokens = self.cls_token.expand((B * T, -1, -1))  # [1,1,F]->[BT,1,F]
        x = paddle.concat((cls_tokens, x), axis=1)
        pos_interp = (x.shape[1] != self.pos_embed.shape[1])
        if pos_interp:
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(
                (0, 2, 1))
            P = int(other_pos_embed.shape[2]**0.5)
            H = x.shape[1] // W
            other_pos_embed = other_pos_embed.reshape([1, x.shape[2], P, P])
            new_pos_embed = F.interpolate(other_pos_embed,
                                          size=(H, W),
                                          mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose((0, 2, 1))
            new_pos_embed = paddle.concat((cls_pos_embed, new_pos_embed),
                                          axis=1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        # Attention blocks
        for blk in self.blocks:
            x = blk(x)


        x = self.norm(x)
        return x[:, 0]  # [B,  embed_dim]  -> [B*T, embed_dim]

    def forward(self, x):
        x = self.forward_features(x)
        return x