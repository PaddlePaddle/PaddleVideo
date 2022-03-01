# Copyright 2021 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import time
import itertools

import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

from paddle import Tensor
from typing import Optional
from collections import OrderedDict

from base import BaseModel
from model.net_vlad import NetVLAD
try:
    from paddlenlp.transformers import BertModel
except ImportError as e:
    print(
        f"{e}, [paddlenlp] package and it's dependencies is required for T2VLAD."
    )


class Mish(nn.Layer):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    SRC: https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py
    '''
    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return input * paddle.tanh(F.softplus(input))


def kronecker_prod(t1, t2):
    # kronecker is performed along the last dim
    kron = paddle.bmm(t1.reshape([-1, t1.size(-1)], 1),
                      t2.reshape([-1, 1, t2.size(-1)]))
    return kron.reshape[(t1.shape[0], t1.shape[1], -1)]


def drop_nans(x, ind, validate_missing):
    """Remove nans, which we expect to find at missing indices.
    Args:
        x (paddle.Tensor): features
        ind (paddle.Tensor): binary values denoting whether or not a given feature is present
        validate_missing (bool): whether to validate that the missing location contains a nan.

    Returns:
        (paddle.tensor): the features, with the missing values masked to zero.
    """

    missing = paddle.nonzero(ind == 0).flatten()
    if missing.numel():
        if validate_missing:
            vals = x[missing[0]]
            assert paddle.isnan(vals.reshape(
                [-1])[0]), "expected nans at missing locations"
        #Prevent overwrite of the original tensor
        x_ = x
        x_[missing] = 0
        x = x_
    if paddle.isnan(x).sum() > 0:
        raise ValueError("Still find nans after removing it!")
    return x


class CENet(BaseModel):
    def __init__(self, text_dim, expert_dims, vlad_clusters, ghost_clusters,
                 feat_aggregation, ce_shared_dim, use_mish, mimic_ce_dims):
        super().__init__()
        self.expert_dims = expert_dims
        self.feat_aggregation = feat_aggregation

        vlad_feat_sizes = {key: val for key, val in vlad_clusters.items()}

        if vlad_clusters["text"] == 0:
            self.text_pooling = nn.Sequential()
        else:
            self.text_pooling = NetVLAD(
                feature_size=text_dim,
                cluster_size=vlad_clusters["text"],
                ghost_clusters=ghost_clusters["text"],
            )
            self.text_bert = BertModel.from_pretrained('bert-base-uncased')
            text_dim = self.text_pooling.out_dim

        self.ce = CEModule(
            text_dim=text_dim,
            expert_dims=expert_dims,
            vlad_feat_sizes=vlad_feat_sizes,
            mimic_ce_dims=mimic_ce_dims,
            use_mish=use_mish,
            same_dim=ce_shared_dim,
        )

    def forward(self,
                experts,
                ind,
                cap_id=None,
                att_mask=None,
                text=None,
                raw_captions=None,
                text_token_mask=None):
        aggregated_experts = OrderedDict()

        # Handle all nan-checks
        for mod in self.expert_dims:
            experts[mod] = drop_nans(x=experts[mod],
                                     ind=ind[mod],
                                     validate_missing=True)
            aggregated_experts[mod] = experts[mod]

        start = time.time()
        # When pooling multiple captions for a single video, we treat them as separate
        # members of the minibatch, so the total pooling op does the following:
        # pooling: B x captions_per_video x max_sentence_length x text_feat_dim
        # -> B x captions_per_video (cluster_dim * text_feat_dim)
        B, captions_per_video, max_words, text_feat_dim = text.shape
        text = text.reshape([B * captions_per_video, max_words, text_feat_dim])
        if isinstance(self.text_pooling, NetVLAD):
            kwargs = {"mask": text_token_mask}
        else:
            kwargs = {}
        cap_id = cap_id.reshape([B * captions_per_video, -1])
        att_mask = att_mask.reshape([B * captions_per_video, -1])
        att_mask = att_mask.unsqueeze(axis=[1, 2])
        bert_out = self.text_bert(cap_id,
                                  token_type_ids=None,
                                  attention_mask=att_mask)
        text = bert_out[0]
        text, _, save_ass = self.text_pooling(text, **kwargs)
        text = text.reshape([B, captions_per_video, -1])

        return self.ce(text, aggregated_experts, ind, raw_captions,
                       self.text_pooling, start)


def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for i in range(N)])


class TransformerLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=True):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        q = q.transpose([1, 0, 2])
        k = k.transpose([1, 0, 2])
        src = src.transpose([1, 0, 2])
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src2 = src2.transpose([1, 0, 2])
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self,
                    src,
                    src_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        q = q.transpose([1, 0, 2])
        k = k.transpose([1, 0, 2])
        src2 = src2.transpose([1, 0, 2])
        src2 = self.self_attn(q, key=k, value=src2, attn_mask=src_mask)
        src2 = src2.transpose([1, 0, 2])
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, pos)
        return self.forward_post(src, src_mask, pos)


class Transformer(nn.Layer):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():  # may have a problem
            if p.dim() > 1:
                nn.initializer.XavierUniform(p)

    def forward(self,
                src,
                mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output)

        if self.norm is not None:
            output = self.norm(output)

        return output


class CEModule(nn.Layer):
    def __init__(self, expert_dims, text_dim, use_mish, mimic_ce_dims,
                 vlad_feat_sizes, same_dim):
        super().__init__()

        modalities = list(expert_dims.keys())
        self.expert_dims = expert_dims
        self.modalities = modalities
        self.mimic_ce_dims = mimic_ce_dims
        self.same_dim = same_dim
        self.use_mish = use_mish
        self.vlad_feat_sizes = vlad_feat_sizes
        self.reduce_dim = 64
        self.moe_cg = ContextGating
        self.vis_transformer = True

        if self.use_mish:
            self.non_lin = Mish()
        else:
            self.non_lin = nn.ReLU()

        num_mods = len(expert_dims)
        self.moe_fc = nn.Linear(text_dim, len(expert_dims))
        self.moe_weights = paddle.ones([1, num_mods]) / num_mods

        # The batch size of the face input can vary (due to missing inputs), so we
        # probably shouldn't use BN on this branch. It's probably fine to leave it
        # n for the corresponding text inputs, (but we should switch to GN)
        use_bns = [True for modality in self.modalities]

        # NOTE: When use_ce is not used, the text features are projected to
        # subspaces of different dimensions.  When use_ce is used, they must all
        # be projected to `same_dim` (to allow fusion). The only excpetion is for an
        # ablation in which we mimic the `same_dim` reduction to measure whether this
        # projection influences overall performance.

        self.repeat_temporal = {}
        for mod in modalities:
            self.repeat_temporal[mod] = 1

        in_dims = [
            expert_dims[mod][0] * self.repeat_temporal[mod]
            for mod in modalities
        ]
        agg_dims = [
            expert_dims[mod][1] * self.repeat_temporal[mod]
            for mod in modalities
        ]
        feat_dims = [
            expert_dims[mod][0] // self.vlad_feat_sizes[mod]
            for mod in modalities
        ]
        if self.vis_transformer:
            num_encoder_layers = 1
            d_model = 768
            nhead = 4
            dim_feedforward = 768
            dropout = 0  #dropout=0.1
            normalize_before = True
            encoder_layer = TransformerLayer(d_model, nhead, dim_feedforward,
                                             dropout)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.transformers = Transformer(encoder_layer, num_encoder_layers,
                                            encoder_norm)

        if self.mimic_ce_dims:
            dim_reducers = [ReduceDim(in_dim, same_dim) for in_dim in feat_dims]
            self.video_dim_reduce = nn.LayerList(dim_reducers)

        gated_vid_embds = [
            GatedEmbeddingUnit(in_dim, same_dim, use_bn=True)
            for in_dim in feat_dims
        ]
        text_out_dims = [same_dim for _ in agg_dims]
        self.video_GU = nn.LayerList(gated_vid_embds)
        gated_text_embds = [
            GatedEmbeddingUnit(text_dim, dim, use_bn=True)
            for dim in text_out_dims
        ]
        self.text_GU = nn.LayerList(gated_text_embds)

    def compute_moe_weights(self, text, ind):
        # compute weights for all captions (including when assigned K captions to
        # the same video)
        B, K, D = text.shape
        M = len(self.modalities)
        msg = f"expected between 1 and 10 modalities, found {M} ({self.modalities})"
        assert 1 <= M <= 10, msg

        # Treat each caption independently in the softmax (which runs over modalities)
        text = text.reshape([B * K, D])

        moe_weights = self.moe_fc(text)  # BK x D -> BK x M
        moe_weights = F.softmax(moe_weights, axis=1)
        moe_weights = moe_weights.reshape([B, K, M])
        return moe_weights

    def forward(self, text, experts, ind, raw_captions, vis_vlad, stime):
        """Compute joint embeddings and, if requested, a confusion matrix between
        video and text representations in the minibatch.

        Notation: B = batch size, M = number of modalities
        """

        # Pass text embeddings through gated units
        text_embd = {}

        # Unroll repeated captions into present minibatch
        B, captions_per_video, feat_dim = text.shape
        text = text.reshape([B * captions_per_video, feat_dim])
        for modality, layer in zip(self.modalities, self.text_GU):
            # NOTE: Due to the batch norm, the gated units are sensitive to passing
            # in a lot of zeroes, so we do the masking step after the forwards pass
            text_ = layer(text)

            # We always assume that text is available for retrieval
            text_ = text_.reshape([B, captions_per_video, -1])
            text_embd[modality] = text_
        text = text.reshape([B, captions_per_video, -1])

        # vladded nans are handled earlier (during pooling)
        # We also avoid zeroing random features, since this will leak information
        # exclude = list(self.vlad_feat_sizes.keys()) + list(self.random_feats)
        # experts = self.mask_missing_embeddings(experts, ind, exclude=exclude)

        # MOE weights computation + normalization - note that we use the first caption
        # sample to predict the weights
        moe_weights = self.compute_moe_weights(text, ind=ind)
        text_local = text.reshape([B * captions_per_video, -1])

        vis_local = {}
        for modality in self.modalities:
            vis_local[modality] = experts[modality]

        all_vis_feat = []
        if hasattr(self, "video_dim_reduce"):
            # Embed all features to a common dimension
            for modality, layer in zip(self.modalities, self.video_dim_reduce):
                all_vis_feat.append(layer(vis_local[modality]))
        all_vis_feat = paddle.concat(all_vis_feat, axis=1)

        if self.vis_transformer:
            experts_tensor = all_vis_feat
            experts_tensor = experts_tensor.transpose([1, 0, 2])
            att_out = self.transformers(experts_tensor, mask=None, pos=None)
            all_vis_feat = att_out.transpose([1, 0, 2])

        vis_local, _, save_ass = vis_vlad(all_vis_feat, freeze=True)
        cross_view_conf_matrix_tv = paddle.matmul(text_local, vis_local.t())

        for modality in self.modalities:
            experts[modality] = experts[modality].max(axis=1)

        for modality, layer in zip(self.modalities, self.video_GU):
            experts[modality] = layer(experts[modality])

        cross_view_conf_matrix = sharded_cross_view_inner_product(
            ind=ind,
            vid_embds=experts,
            text_embds=text_embd,
            text_weights=moe_weights,
            subspaces=self.modalities,
            raw_captions=raw_captions,
        )
        cross_view_conf_matrix = 0.5 * cross_view_conf_matrix + 0.5 * cross_view_conf_matrix_tv
        return {
            "modalities": self.modalities,
            "cross_view_conf_matrix": cross_view_conf_matrix,
        }


class GatedEmbeddingUnit(nn.Layer):
    def __init__(self, input_dimension, output_dimension, use_bn):
        super(GatedEmbeddingUnit, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension, add_batch_norm=use_bn)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        x = F.normalize(x)
        return x


class ReduceDim(nn.Layer):
    def __init__(self, input_dimension, output_dimension):
        super(ReduceDim, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x, axis=-1)
        return x


class ContextGating(nn.Layer):
    def __init__(self, dimension, add_batch_norm=True):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1D(dimension)

    def forward(self, x):
        x1 = self.fc(x)
        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
        x = paddle.concat([x, x1], axis=1)
        return F.glu(x, axis=1)


def sharded_cross_view_inner_product(vid_embds,
                                     text_embds,
                                     text_weights,
                                     subspaces,
                                     ind,
                                     tol=1E-5,
                                     raw_captions=None):
    """Compute a similarity matrix from sharded vectors.

    Args:
        embds1 (dict[str:paddle.Tensor]): the set of sub-embeddings that, when
            concatenated, form the whole. The ith shard has shape `B x K x F_i`
            (i.e. they can differ in the last dimension).
        embds2 (dict[str:paddle.Tensor]): same format.
        weights2 (paddle.Tensor): weights for the shards in `embds2`.

    Returns:
        (paddle.tensor): similarity matrix of size `BK x BK`.

    NOTE: If multiple captions are provided, we can aggregate their similarities to
    provide a single video-text similarity score.
    """
    B = vid_embds[subspaces[0]].shape[0]
    T, num_caps, _ = text_embds[subspaces[0]].shape

    # unroll separate captions onto first dimension and treat them separately
    sims = paddle.zeros([T * num_caps, B])
    text_weights = text_weights.reshape([T * num_caps, -1])
    if True:
        mus = [round(x, 3) for x in text_weights.mean(0).numpy().tolist()]
        stds = [round(x, 3) for x in text_weights.std(0).numpy().tolist()]
        summary = ">>>"
        for mod, mu, std in zip(subspaces, mus, stds):
            summary += f"{mod}: {mu} +/- {std} "

    # mark expert availabilities along the second axis
    available = paddle.ones([1, B, len(subspaces)], dtype=text_weights.dtype)
    for ii, modality in enumerate(subspaces):
        ind[modality] = paddle.to_tensor(ind[modality], dtype='float32')
        available[:, :, ii] = ind[modality]
    msg = "expected `available` modality mask to only contain 0s or 1s"
    assert set(paddle.unique(available).cpu().numpy()).issubset(set([0,
                                                                     1])), msg
    # set the text weights along the first axis and combine with availabilities to
    # produce a <T x B x num_experts> tensor
    text_weight_tensor = text_weights.reshape([T * num_caps, 1,
                                               len(subspaces)]) * available
    # normalise to account for missing experts
    normalising_weights = text_weight_tensor.sum(2).reshape(
        [T * num_caps, B, 1])
    text_weight_tensor = paddle.divide(text_weight_tensor, normalising_weights)

    l2_mass_text, l2_mass_vid = 1, 1

    for idx, modality in enumerate(subspaces):
        vid_embd_ = vid_embds[modality].reshape([B, -1]) / l2_mass_vid
        text_embd_ = text_embds[modality].reshape([T * num_caps, -1])
        msg = "expected weights to be applied to text embeddings"
        assert text_embd_.shape[0] == text_weights.shape[0], msg
        text_embd_ = text_embd_ / l2_mass_text
        weighting = text_weight_tensor[:, :, idx]
        sims += weighting * paddle.matmul(text_embd_,
                                          vid_embd_.t())  # (T x num_caps) x (B)

    if paddle.isnan(sims).sum().item():
        raise ValueError("Found nans in similarity matrix!")

    return sims
