"""NetVLAD implementation.
"""
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

import math
import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F


class NetVLAD(nn.Layer):
    def __init__(self, cluster_size, feature_size, ghost_clusters=0,
                 add_batch_norm=True):
        super().__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        init_sc = paddle.to_tensor(init_sc)
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = paddle.create_parameter([feature_size, clusters], dtype='float32', default_initializer=nn.initializer.Assign(paddle.randn([feature_size, clusters]) * init_sc))
        self.batch_norm1 = nn.BatchNorm1D(clusters) if add_batch_norm else None
        self.batch_norm2 = nn.BatchNorm1D(clusters) if add_batch_norm else None
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters1 = paddle.create_parameter([1, feature_size, cluster_size], dtype='float32', default_initializer=nn.initializer.Assign(paddle.randn([1, feature_size, cluster_size]) * init_sc))
        self.clusters2 = paddle.create_parameter([1, feature_size, cluster_size], dtype='float32', default_initializer=nn.initializer.Assign(paddle.randn([1, feature_size, cluster_size]) * init_sc)) 
        self.out_dim = self.cluster_size * feature_size
    
    def sanity_checks(self, x):
        """Catch any nans in the inputs/clusters"""
        if paddle.isnan(paddle.sum(x)):
            raise ValueError("nan inputs")
        if paddle.isnan(self.clusters[0][0]): 
            raise ValueError("nan clusters")
        
    def forward(self, x, freeze=False, mask=None):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (th.Tensor): B x N x D

        Returns:
            (th.Tensor): B x DK
        """
        self.sanity_checks(x)
        max_sample = x.shape[1] 
        x = x.reshape([-1, self.feature_size]) # B x N x D -> BN x D

        if freeze == True:
            clusters = self.clusters.detach()
            clusters2 = self.clusters1
            batch_norm =  self.batch_norm1
        else:
            clusters = self.clusters
            clusters2 = self.clusters2
            batch_norm =  self.batch_norm2

        assignment = paddle.matmul(x, clusters) # (BN x D) x (D x (K+G)) -> BN x (K+G)
        if batch_norm:
            assignment = batch_norm(assignment)

        assignment = F.softmax(assignment, axis=1) # BN x (K+G) -> BN x (K+G)
        save_ass = assignment.reshape([-1, max_sample, self.cluster_size+1])

        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.reshape([-1, max_sample, self.cluster_size]) # -> B x N x K
        a_sum = paddle.sum(assignment, axis=1, keepdim=True) # B x N x K -> B x 1 x K
        a = a_sum * self.clusters2
        assignment = assignment.transpose([0, 2, 1])  # B x N x K -> B x K x N

        x = x.reshape([-1, max_sample, self.feature_size]) # BN x D -> B x N x D
        vlad = paddle.matmul(assignment, x) # (B x K x N) x (B x N x D) -> B x K x D
        vlad = vlad.transpose([0, 2, 1]) # -> B x D x K
        vlad = vlad - a

        # L2 intra norm
        vlad_ = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad_.reshape([-1, self.cluster_size * self.feature_size])  # -> B x DK
        vlad = F.normalize(vlad)
        return vlad, vlad_, save_ass  # B x DK