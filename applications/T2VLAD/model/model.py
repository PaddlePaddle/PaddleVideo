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
from utils import expert_tensor_storage
from paddlenlp.transformers import BertModel


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
    kron = paddle.bmm(t1.reshape([-1, t1.size(-1)], 1), t2.reshape([-1, 1, t2.size(-1)]))
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
            assert paddle.isnan(vals.reshape([-1])[0]), "expected nans at missing locations"
        #Prevent overwrite of the original tensor
        x_ = x
        x_[missing] = 0
        x = x_
    if paddle.isnan(x).sum() > 0:
        raise ValueError('nan exists')
    return x

class CENet(BaseModel):
    def __init__(
            self,
            task,
            use_ce,
            text_dim,
            l2renorm,
            expert_dims,
            vlad_clusters,
            ghost_clusters,
            disable_nan_checks,
            keep_missing_modalities,
            test_caption_mode,
            randomise_feats,
            feat_aggregation,
            ce_shared_dim,
            trn_config,
            trn_cat,
            include_self,
            use_mish,
            use_bn_reason,
            num_h_layers,
            num_g_layers,
            kron_dets=False,
            freeze_weights=False,
            geometric_mlp=False,
            rand_proj=False,
            mimic_ce_dims=False,
            coord_dets=False,
            concat_experts=False,
            spatial_feats=False,
            concat_mix_experts=False,
            verbose=False,
            num_classes=None):
        super().__init__()

        self.l2renorm = l2renorm
        self.task = task
        self.geometric_mlp = geometric_mlp
        self.feat_aggregation = feat_aggregation
        self.expert_dims = expert_dims
        self.num_h_layers = num_h_layers
        self.num_g_layers = num_g_layers
        self.use_mish = use_mish
        self.use_bn_resaon = use_bn_reason
        self.include_self = include_self
        self.kron_dets = kron_dets
        self.rand_proj = rand_proj
        self.coord_dets = coord_dets
        self.disable_nan_checks = disable_nan_checks
        self.trn_config = trn_config
        self.trn_cat = trn_cat
        if randomise_feats:
            self.random_feats = set([x for x in randomise_feats.split(",")])
        else:
            self.random_feats = set()

        # sanity checks on the features that may be vladded
        pre_vlad_feat_sizes = {"ocr": 300, "audio": 128, "speech": 300}
        pre_vlad_feat_sizes = {key: val for key, val in pre_vlad_feat_sizes.items()
                               if feat_aggregation[key]["temporal"] == "vlad"}

        # we basically disable safety checks for detection-sem
        if spatial_feats:
            spatial_feat_dim = 16
        else:
            spatial_feat_dim = 5
        if self.geometric_mlp:
            self.geometric_mlp_model = SpatialMLP(spatial_feat_dim)
        if kron_dets:
            sem_det_dim = 300 * spatial_feat_dim
        elif coord_dets:
            sem_det_dim = spatial_feat_dim
        elif rand_proj:
            sem_det_dim = 300 + 300
            self.proj = nn.Linear(spatial_feat_dim, 300)
        else:
            sem_det_dim = 300 + spatial_feat_dim
        self.spatial_feat_dim = spatial_feat_dim
        pre_vlad_feat_sizes["detection-sem"] = sem_det_dim
        if "detection-sem" in expert_dims:
            new_in_dim = sem_det_dim * vlad_clusters["detection-sem"]
            expert_dims["detection-sem"] = (new_in_dim, expert_dims["detection-sem"][1])

        vlad_feat_sizes = {key: val for key, val in vlad_clusters.items()}

        self.pooling = nn.LayerDict() 
        for mod, expected in pre_vlad_feat_sizes.items():
            if mod in expert_dims.keys():
                feature_size = expert_dims[mod][0] // vlad_clusters[mod]
                msg = f"expected {expected} for {mod} features atm"
                assert feature_size == expected, msg
                self.pooling[mod] = NetVLAD(
                    feature_size=feature_size,
                    cluster_size=vlad_clusters[mod],
                )
        if "retrieval" in self.task:
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
        else:
            self.num_classes = num_classes
            text_dim = None

        self.tensor_storage = expert_tensor_storage(
            experts=self.expert_dims.keys(),
            feat_aggregation=self.feat_aggregation,
        )

        self.ce = CEModule(
            use_ce=use_ce,
            task=self.task,
            verbose=verbose,
            l2renorm=l2renorm,
            trn_cat=self.trn_cat,
            trn_config=self.trn_config,
            random_feats=self.random_feats,
            freeze_weights=freeze_weights,
            text_dim=text_dim,
            test_caption_mode=test_caption_mode,
            concat_experts=concat_experts,
            concat_mix_experts=concat_mix_experts,
            expert_dims=expert_dims,
            vlad_feat_sizes=vlad_feat_sizes,
            disable_nan_checks=disable_nan_checks,
            keep_missing_modalities=keep_missing_modalities,
            mimic_ce_dims=mimic_ce_dims,
            include_self=include_self,
            use_mish=use_mish,
            use_bn_reason=use_bn_reason,
            num_h_layers=num_h_layers,
            num_g_layers=num_g_layers,
            num_classes=num_classes,
            same_dim=ce_shared_dim,
        )

    def randomise_feats(self, experts, key):
        if key in self.random_feats:
            # keep expected nans
            nan_mask = paddle.isnan(experts[key])
            experts[key] = paddle.randn(experts[key].shape, dtype=experts[key].dtype) 
            if not self.disable_nan_checks:    
                nans = paddle.to_tensor(float('nan')) 
                experts[key][nan_mask] = nans 
        return experts

    def forward(self, experts, ind, cap_id=None, att_mask=None, text=None, raw_captions=None, text_token_mask=None):
        aggregated_experts = OrderedDict()

        if "detection-sem" in self.expert_dims:
            det_sem = experts["detection-sem"]
            box_feats = det_sem[:, :, :self.spatial_feat_dim]
            sem_feats = det_sem[:, :, self.spatial_feat_dim:]
            if self.geometric_mlp:
                x = box_feats.reshape[(-1, box_feats.shape[-1])]
                x = self.geometric_mlp_model(x)
                box_feats = x.reshape[(box_feats.shape)]

            if self.kron_dets:
                feats = kronecker_prod(box_feats, sem_feats)
            elif self.coord_dets:
                feats = box_feats
            elif self.rand_proj:
                feats = box_feats
                projected = self.proj(feats)
                feats = paddle.concat([projected, sem_feats], axis=2) 
            else:
                feats = paddle.concat([projected, sem_feats], axis=2)
            experts["detection-sem"] = feats

        # Handle all nan-checks
        for mod in self.expert_dims:
            experts = self.randomise_feats(experts, mod)
            experts[mod] = drop_nans(x=experts[mod], ind=ind[mod], validate_missing=True)
            if mod not in self.pooling.keys():
                aggregated_experts[mod] = experts[mod]
            else:
                aggregated_experts[mod] = self.pooling[mod](experts[mod])

        start = time.time()
        if "retrieval" in self.task:
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
            att_mask = att_mask.unsqueeze(axis=[1,2])
            bert_out = self.text_bert(cap_id, token_type_ids=None, attention_mask=att_mask)
            text = bert_out[0]
            text, _, save_ass = self.text_pooling(text, **kwargs)
            text = text.reshape([B, captions_per_video, -1])
            
        else:
            text = None
        return self.ce(text, aggregated_experts, ind, raw_captions, self.text_pooling, start)

class TemporalAttention(nn.Layer):
    def __init__(self, img_feature_dim, num_attention):
        super().__init__()
        self.weight = paddle.randn([img_feature_dim, num_attention])
        self.img_feature_dim = img_feature_dim
        self.num_attention = num_attention

    def forward(self, input):
        B, T, D = input.shape
        record = []
        input_avg = paddle.mean(input.clone(), axis=1) 
        input_max = paddle.max(input.clone(), axis=1) 
        record.append(input_avg)
        record.append(input_max[0])
        output = paddle.matmul(input, self.weight) 
        attentions = F.softmax(output, axis=1) 
        for idx in range(attentions.shape[-1]):
            temp = attentions[:, :, idx]
            temp_output = paddle.sum(temp.unsqueeze(2) * input, axis=1) 
            norm = temp_output.norm(p=2, axis=-1, keepdim=True)
            temp_output = temp_output.divide(norm)
            record.append(temp_output)
        act_all = paddle.concat([record], axis=1)
        return act_all

class RelationModuleMultiScale(nn.Layer):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScale_Cat, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.LayerList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Linear(num_bottleneck, self.num_class),
                        )

            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        record = []
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.reshape([act_all.size(0), self.scales[0] * self.img_feature_dim])
        act_all = self.fc_fusion_scales[0](act_all)
        norm = act_all.norm(p=2, axis=-1, keepdim=True)
        act_all = act_all.divide(norm)
        record.append(act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            act_all = 0
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.reshape([act_relation.size(0), self.scales[scaleID] * self.img_feature_dim])
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
            norm = act_all.norm(p=2, axis=-1, keepdim=True)
            act_all = act_all.divide(norm)
            record.append(act_all)

        act_all = paddle.concat([record], axis=1)
        return act_all  

    def return_relationset(self, num_frames, num_frames_relation):
        return list(itertools.combinations([i for i in range(num_frames)], 
                    num_frames_relation)) 

class RelationModuleMultiScale_Cat(nn.Layer):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScale_Cat, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.LayerList()
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Linear(num_bottleneck, self.num_class),
                        )

            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        record = []
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.reshape([act_all.shape[0], self.scales[0] * self.img_feature_dim]) 
        act_all = self.fc_fusion_scales[0](act_all)
        norm = act_all.norm(p=2, axis=-1, keepdim=True) 
        act_all = act_all.divide(norm)
        record.append(act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            act_all = 0
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.reshape([act_relation.shape[0], self.scales[scaleID] * self.img_feature_dim])
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
            norm = act_all.norm(p=2, axis=-1, keepdim=True)
            act_all = act_all.divide(norm)
            record.append(act_all)

        act_all = paddle.concat([record], 1)
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        return list(itertools.combinations([i for i in range(num_frames)], 
                    num_frames_relation))

def _get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for i in range(N)]) 

class TransformerLayer(nn.Layer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True):
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

    def forward_post(self, src,
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
    
    def forward_pre(self, src,
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

    def forward(self, src,
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
        for p in self.parameters(): # may have a problem
            if p.dim() > 1:
                nn.initializer.XavierUniform(p) 

    def forward(self, src,
                mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output)

        if self.norm is not None:
            output = self.norm(output)

        return output

class CEModule(nn.Layer):
    def __init__(self, expert_dims, text_dim, use_ce, verbose, l2renorm, num_classes,
                 trn_config, trn_cat, use_mish, include_self, num_h_layers, num_g_layers,
                 disable_nan_checks, random_feats, test_caption_mode, mimic_ce_dims,
                 concat_experts, concat_mix_experts, freeze_weights, task,
                 keep_missing_modalities, vlad_feat_sizes, same_dim, use_bn_reason):
        super().__init__()

        modalities = list(expert_dims.keys())
        self.expert_dims = expert_dims
        self.modalities = modalities
        self.disable_nan_checks = disable_nan_checks
        self.mimic_ce_dims = mimic_ce_dims
        self.concat_experts = concat_experts
        self.same_dim = same_dim
        self.use_mish = use_mish
        self.use_bn_reason = use_bn_reason
        self.num_h_layers = num_h_layers
        self.num_g_layers = num_g_layers
        self.include_self = include_self
        self.num_classes = num_classes
        self.task = task
        self.vlad_feat_sizes = vlad_feat_sizes
        self.concat_mix_experts = concat_mix_experts
        self.test_caption_mode = test_caption_mode
        self.reduce_dim = 64
        self.moe_cg = ContextGating
        self.freeze_weights = freeze_weights
        self.random_feats = random_feats
        self.use_ce = use_ce
        self.verbose = verbose
        self.keep_missing_modalities = keep_missing_modalities
        self.l2renorm = l2renorm
        self.trn_config = trn_config
        self.trn_cat = trn_cat
        self.vis_transformer = True

        if self.use_mish:
            self.non_lin = Mish()
        else:
            self.non_lin = nn.ReLU()

        if "retrieval" in self.task:
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
        self.trn_list = nn.LayerList()

        self.repeat_temporal = {}
        for mod in modalities:
            self.repeat_temporal[mod] = 1

        if self.trn_cat == 2:
            print("Performing concat between random temporal attention")
            for mod in self.trn_config.keys():
                img_feature_dim = expert_dims[mod][0]  # 365
                num_frames = self.trn_config[
                    mod]  # This is exatcly how many different attention
                num_frames = 1  # mimic simple avg and max based on segments
                self.trn_list += [TemporalAttention(img_feature_dim, num_frames)]
                self.repeat_temporal[mod] = num_frames + 2
        elif self.trn_cat == 1:
            print("Performing concat between segments")
            for mod in self.trn_config.keys():
                img_feature_dim = expert_dims[mod][0]  # 365
                num_frames = self.trn_config[mod]  # hard code
                num_class = expert_dims[mod][0]
                self.trn_list += [
                    RelationModuleMultiScale_Cat(img_feature_dim, num_frames, num_class)
                ]
                self.repeat_temporal[mod] = len(
                    [i for i in range(num_frames, 1, -1)])
        elif self.trn_cat == 0:
            print("Performing Conventional TRN (sum) segments")
            for mod in self.trn_config.keys():
                img_feature_dim = expert_dims[mod][0]  # 365
                num_frames = self.trn_config[mod]  # hard code
                num_class = expert_dims[mod][0]
                self.trn_list += [
                    RelationModuleMultiScale(img_feature_dim, num_frames,
                                             num_class)
                ]
        else:
            raise NotImplementedError()

        in_dims = [expert_dims[mod][0] * self.repeat_temporal[mod] for mod in modalities]
        agg_dims = [expert_dims[mod][1] * self.repeat_temporal[mod] for mod in modalities]
        feat_dims = [expert_dims[mod][0] // self.vlad_feat_sizes[mod] for mod in modalities]
        if self.vis_transformer:
            num_encoder_layers = 1
            d_model = 768
            nhead = 4
            dim_feedforward = 768
            dropout=0 #dropout=0.1
            normalize_before=True
            encoder_layer = TransformerLayer(d_model, nhead, dim_feedforward,dropout)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.transformers = Transformer(encoder_layer, num_encoder_layers, encoder_norm)

        if self.use_ce or self.mimic_ce_dims:
            dim_reducers = [ReduceDim(in_dim, same_dim) for in_dim in feat_dims]
            self.video_dim_reduce = nn.LayerList(dim_reducers)

        if self.use_ce:
            # The g_reason module has a first layer that is specific to the design choice
            # (e.g. triplet vs pairwise), then a shared component which is common to all
            # designs.
            if self.use_ce in {"pairwise", "pairwise-star", "triplet"}:
                num_inputs = 3 if self.use_ce == "triplet" else 2
                self.g_reason_1 = nn.Linear(same_dim * num_inputs, same_dim)
            elif self.use_ce == "pairwise-star-specific":
                num_inputs = 2
                g_reason_unshared_weights = [G_reason(same_dim, num_inputs, self.non_lin)
                                             for mod in modalities]
                self.g_reason_unshared_weights = nn.LayerList(g_reason_unshared_weights)
            elif self.use_ce in {"pairwise-star-tensor"}:
                reduce_dim = self.reduce_dim
                self.dim_reduce = nn.Linear(same_dim, reduce_dim)
                self.g_reason_1 = nn.Linear(self.reduce_dim * reduce_dim, same_dim)
            else:
                raise ValueError(f"unrecognised CE config: {self.use_ce}")

            g_reason_shared = []
            for _ in range(self.num_g_layers - 1):
                if self.use_bn_reason:
                    g_reason_shared.append(nn.BatchNorm1D(same_dim, momentum=0.1))
                g_reason_shared.append(self.non_lin)
                g_reason_shared.append(nn.Linear(same_dim, same_dim))
            self.g_reason_shared = nn.Sequential(*g_reason_shared)

            h_reason = []
            for _ in range(self.num_h_layers):
                if self.use_bn_reason:
                    h_reason.append(nn.BatchNorm1D(same_dim))
                h_reason.append(self.non_lin)
                h_reason.append(nn.Linear(same_dim, same_dim))
            self.h_reason = nn.Sequential(*h_reason)

            gated_vid_embds = [GatedEmbeddingUnitReasoning(same_dim) for _ in in_dims]
            text_out_dims = [same_dim for _ in agg_dims]
        elif self.mimic_ce_dims:  # ablation study

            gated_vid_embds = [GatedEmbeddingUnit(in_dim, same_dim, use_bn=True)
                               for in_dim in feat_dims]
            text_out_dims = [same_dim for _ in agg_dims]
        elif self.concat_mix_experts:  # ablation study
            # use a single large GEU to mix the experts - the output will be the sum
            # of the aggregation sizes
            in_dim, out_dim = sum(in_dims), sum(agg_dims)
            gated_vid_embds = [GatedEmbeddingUnit(in_dim, out_dim, use_bn=True)]
        elif self.concat_experts:  # ablation study
            # We do not use learnable parameters for the video combination, (we simply
            # use a high dimensional inner product).
            gated_vid_embds = []
        else:
            gated_vid_embds = [GatedEmbeddingUnit(in_dim, dim, use_bn) for
                               in_dim, dim, use_bn in zip(in_dims, agg_dims, use_bns)]
            text_out_dims = agg_dims
        self.video_GU = nn.LayerList(gated_vid_embds)

        if "retrieval" in self.task:
            if self.concat_experts:
                gated_text_embds = [nn.Sequential()]
            elif self.concat_mix_experts:
                # As with the video inputs, we similiarly use a single large GEU for the
                # text embedding
                gated_text_embds = [GatedEmbeddingUnit(text_dim, sum(agg_dims),
                                    use_bn=True)]
            else:
                gated_text_embds = [GatedEmbeddingUnit(text_dim, dim, use_bn=True) for
                                    dim in text_out_dims]
            self.text_GU = nn.LayerList(gated_text_embds)
        else:
            print("V. simple classifier, should update....")
            total_dim = 0
            for mod in self.expert_dims.keys():
                total_dim += self.expert_dims[mod][1] * self.repeat_temporal[mod]
            print(f"Total dim is {total_dim}")
            self.classifier = nn.Linear(total_dim, self.num_classes)

    def compute_moe_weights(self, text, ind):
        # compute weights for all captions (including when assigned K captions to
        # the same video)
        B, K, D = text.shape
        M = len(self.modalities)
        msg = f"expected between 1 and 10 modalities, found {M} ({self.modalities})"
        assert 1 <= M <= 10, msg

        # Treat each caption independently in the softmax (which runs over modalities)
        text = text.reshape([B * K, D]) 
        if self.freeze_weights:
            moe_weights = paddle.to_tensor(np.tile(np.array(self.moe_weights), [B, K, 1])) 
        else:
            moe_weights = self.moe_fc(text)  # BK x D -> BK x M
            moe_weights = F.softmax(moe_weights, axis=1)
            moe_weights = moe_weights.reshape([B, K, M])

        if self.verbose:
            print("--------------------------------")
            for idx, key in enumerate(self.modalities):
                msg = "{}: mean: {:.3f}, std: {:.3f}, min: {:.3f}, max: {:.3f}"
                msg = msg.format(
                    key,
                    moe_weights[:, :, idx].mean().item(),
                    moe_weights[:, :, idx].std().item(),
                    moe_weights[:, :, idx].min().item(),
                    moe_weights[:, :, idx].max().item(),
                )
                print(msg)
        return moe_weights

    def forward(self, text, experts, ind, raw_captions, vis_vlad, stime):
        """Compute joint embeddings and, if requested, a confusion matrix between
        video and text representations in the minibatch.

        Notation: B = batch size, M = number of modalities
        """
        if "retrieval" in self.task:
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

                if "text" in self.random_feats:
                    text_ = paddle.rand(text_.shape) 
                text_embd[modality] = text_
            text = text.reshape([B, captions_per_video, -1])

            # vladded nans are handled earlier (during pooling)
            # We also avoid zeroing random features, since this will leak information
            # exclude = list(self.vlad_feat_sizes.keys()) + list(self.random_feats)
            # experts = self.mask_missing_embeddings(experts, ind, exclude=exclude)

            # MOE weights computation + normalization - note that we use the first caption
            # sample to predict the weights
            moe_weights = self.compute_moe_weights(text, ind=ind)
            text_local = text.reshape([B*captions_per_video, -1])

        if self.l2renorm:
            for modality in self.modalities:
                norm = experts[modality].norm(p=2, axis=-1, keepdim=True)
                experts[modality] = experts[modality].divide(norm)

        for modality, layer in zip(self.modalities, self.trn_list):
            experts[modality] = layer(experts[modality])

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

        if self.training:
            merge_caption_similiarities = "avg"
        else:
            merge_caption_similiarities = self.test_caption_mode

        cross_view_conf_matrix = sharded_cross_view_inner_product(
                ind=ind,
                vid_embds=experts,
                text_embds=text_embd,
                keep_missing_modalities=self.keep_missing_modalities,
                l2renorm=self.l2renorm,
                text_weights=moe_weights,
                subspaces=self.modalities,
                raw_captions=raw_captions,
                merge_caption_similiarities=merge_caption_similiarities,
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

class MimicCEGatedEmbeddingUnit(nn.Layer):
    def __init__(self, input_dimension, output_dimension, use_bn):
        super().__init__()
        self.cg = ContextGating(input_dimension, add_batch_norm=use_bn)

    def forward(self, x):
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

class GatedEmbeddingUnitReasoning(nn.Layer):
    def __init__(self, output_dimension):
        super(GatedEmbeddingUnitReasoning, self).__init__()
        self.cg = ContextGatingReasoning(output_dimension)

    def forward(self, x, mask):
        x = self.cg(x, mask)
        x = F.normalize(x)
        return x

class SpatialMLP(nn.Layer):
    def __init__(self, dimension):
        super().__init__()
        self.cg1 = ContextGating(dimension)
        self.cg2 = ContextGating(dimension)

    def forward(self, x):
        x = self.cg1(x)
        return self.cg2(x)
    
class ContextGatingReasoning(nn.Layer):
    def __init__(self, dimension, add_batch_norm=True):
        super(ContextGatingReasoning, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1D(dimension)
        self.batch_norm2 = nn.BatchNorm1D(dimension)

    def forward(self, x, x1):

        x2 = self.fc(x)

        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
            x2 = self.batch_norm2(x2)

        t = x1 + x2
        x = paddle.concat([x, t], axis=1)
        return F.glu(x, axis=1)

class G_reason(nn.Layer):
    def __init__(self, same_dim, num_inputs, non_lin):
        super().__init__()
        self.g_reason_1_specific = nn.Linear(same_dim * num_inputs, same_dim)
        self.g_reason_2_specific = nn.Linear(same_dim, same_dim)
        self.non_lin = non_lin

    def forward(self, x):
        x = self.g_reason_1_specific(x)  # B x 2D -> B x D
        x = self.non_lin(x)
        x = self.g_reason_2_specific(x)
        return x
        
def sharded_cross_view_inner_product(vid_embds, text_embds, text_weights,
                                     subspaces, l2renorm, ind,
                                     keep_missing_modalities,
                                     merge_caption_similiarities="avg", tol=1E-5,
                                     raw_captions=None):
    """Compute a similarity matrix from sharded vectors.

    Args:
        embds1 (dict[str:paddle.Tensor]): the set of sub-embeddings that, when
            concatenated, form the whole. The ith shard has shape `B x K x F_i`
            (i.e. they can differ in the last dimension).
        embds2 (dict[str:paddle.Tensor]): same format.
        weights2 (paddle.Tensor): weights for the shards in `embds2`.
        l2norm (bool::True): whether to l2 renormalize the full embeddings.

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

    if keep_missing_modalities:
        # assign every expert/text inner product the same weight, even if the expert
        # is missing
        text_weight_tensor = paddle.ones([T * num_caps, B, len(subspaces)], dtype=text_weights.dtype)
    else:
        # mark expert availabilities along the second axis
        available = paddle.ones([1, B, len(subspaces)], dtype=text_weights.dtype)
        for ii, modality in enumerate(subspaces):
            ind[modality] = paddle.to_tensor(ind[modality], dtype='float32')
            available[:, :, ii] = ind[modality]
        msg = "expected `available` modality mask to only contain 0s or 1s"
        assert set(paddle.unique(available).cpu().numpy()).issubset(set([0, 1])), msg 
        # set the text weights along the first axis and combine with availabilities to
        # produce a <T x B x num_experts> tensor
        text_weight_tensor = text_weights.reshape([T*num_caps, 1, len(subspaces)]) * available 
        # normalise to account for missing experts
        normalising_weights = text_weight_tensor.sum(2).reshape([T*num_caps, B, 1]) 
        text_weight_tensor = paddle.divide(text_weight_tensor, normalising_weights)

    if l2renorm:
        raise NotImplementedError("Do not use renorm until availability fix is complete")
        l2_mass_vid, l2_mass_text = 0, 0
        for idx, modality in enumerate(subspaces):
            vid_embd_ = vid_embds[modality]
            assert len(vid_embd_.shape) == 2, "expected B x feat_dim format"
            l2_mass_vid += vid_embd_.reshape([B, -1]).pow(2).sum(1) 
            text_embd_ = text_embds[modality]
            assert len(text_embd_.shape) == 3, "expected B x caps x feat_dim format"
            text_embd_ = text_embd_.reshape([B * num_caps, -1]) 
            text_embd_ = text_weights[:, idx:idx + 1] * text_embd_
            l2_mass_text += text_embd_.pow(2).sum(1)
        l2_mass_vid = paddle.sqrt(l2_mass_vid.clip(min=1E-6)).unsqueeze(1) 
        l2_mass_text = paddle.sqrt(l2_mass_text.clip(min=1E-6)).unsqueeze(1) 
    else:
        l2_mass_text, l2_mass_vid = 1, 1

    for idx, modality in enumerate(subspaces):
        vid_embd_ = vid_embds[modality].reshape([B, -1]) / l2_mass_vid 
        text_embd_ = text_embds[modality].reshape([T * num_caps, -1])
        msg = "expected weights to be applied to text embeddings"
        assert text_embd_.shape[0] == text_weights.shape[0], msg
        text_embd_ = text_embd_ / l2_mass_text
        weighting = text_weight_tensor[:, :, idx]
        sims += weighting * paddle.matmul(text_embd_, vid_embd_.t())  # (T x num_caps) x (B)

    if l2renorm:
        assert sims.max() < 1 + tol, "expected cosine similarities to be < 1"
        assert sims.min() > -1 - tol, "expected cosine similarities to be > -1"

    if paddle.isnan(sims).sum().item(): 
        raise ValueError("Found nans in similarity matrix!")

    if num_caps > 1:
        # aggregate similarities from different captions
        if merge_caption_similiarities == "avg":
            sims = sims.reshape([B, num_caps, B]) 
            sims = paddle.mean(sims, axis=1) 
            sims = sims.reshape[(B, B)] 
        elif merge_caption_similiarities == "indep":
            pass
        else:
            msg = "unrecognised merge mode: {}"
            raise ValueError(msg.format(merge_caption_similiarities))
    return sims

def sharded_single_view_inner_product(embds, subspaces, text_weights=None,
                                      l2renorm=True):
    """Compute a similarity matrix from sharded vectors.

    Args:
        embds (dict[str:paddle.Tensor]): the set of sub-embeddings that, when concatenated,
            form the whole. The ith shard has shape `B x K x F_i` (i.e. they can
            differ in the last dimension), or shape `B x F_i`
        l2norm (bool::True): whether to l2 normalize the full embedding.

    Returns:
        (paddle.tensor): similarity matrix of size `BK x BK`.
    """
    subspaces = list(embds.keys())
    shape = embds[subspaces[0]].shape
    if len(shape) == 3:
        B, K, _ = shape
        num_embds = B * K
        assert text_weights is not None, "Expected 3-dim tensors for text (+ weights)"
        assert text_weights.shape[0] == B
        assert text_weights.shape[1] == K
    elif len(shape) == 2:
        B, _ = shape
        num_embds = B
        assert text_weights is None, "Expected 2-dim tensors for non-text (no weights)"
    else:
        raise ValueError("input tensor with {} dims unrecognised".format(len(shape)))
    sims = paddle.zeros([num_embds, num_embds])
    if l2renorm:
        l2_mass = 0
        for idx, modality in enumerate(subspaces):
            embd_ = embds[modality]
            if text_weights is not None:
                # text_weights (i.e. moe_weights) are shared among subspace for video
                embd_ = text_weights[:, :, idx:idx + 1] * embd_
            embd_ = embd_.reshape([num_embds, -1]) 
            l2_mass += embd_.pow(2).sum(1)
        l2_mass = paddle.sqrt(l2_mass.clip(min=1E-6)).unsqueeze(1) 
    else:
        l2_mass = 1

    for idx, modality in enumerate(subspaces):
        embd_ = embds[modality]
        if text_weights is not None:
            embd_ = text_weights[:, :, idx:idx + 1] * embd_
        embd_ = embd_.reshape([num_embds, -1]) / l2_mass 
        sims += paddle.matmul(embd_, embd_.t())
    if paddle.isnan(sims).sum().item(): 
        raise ValueError("Found nans in similarity matrix!")
    return sims
    