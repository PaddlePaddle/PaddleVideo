"""This module contains an implementation of the max margin ranking loss, slightly
modified from this code:
https://github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/loss.py

The modification is the `fix_norm` conditional, which removes zero terms from the
diagonal when performing the averaging calculation.

Original licence below.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def cosine_sim(im, s):
  '''cosine similarity between all the image and sentence pairs
  '''
  inner_prod = im.mm(s.t())
  im_norm = paddle.sqrt((im ** 2).sum(axis=1).reshape([-1, 1]) + 1e-18) 
  s_norm = paddle.sqrt((s ** 2).sum(axis=1).reshape([-1, 1]) + 1e-18)
  sim = inner_prod / (im_norm * s_norm)
  return sim

class ContrastiveLoss(nn.Layer):
  '''compute contrastive loss
  '''
  def __init__(self, margin=0, max_violation=True, direction='bi', topk=1):
    '''Args:
      direction: i2t for negative sentence, t2i for negative image, bi for both
    '''
    super().__init__()
    self.margin = margin
    self.max_violation = max_violation
    self.direction = direction
    self.topk = topk

  def forward(self, scores, margin=None, average_batch=True):
    '''
    Args:
      scores: image-sentence score matrix, (batch, batch)
        the same row of im and s are positive pairs, different rows are negative pairs
    '''

    if margin is None:
      margin = self.margin

    batch_size = scores.shape[0] 
    diagonal = paddle.diagonal(scores).reshape([batch_size, 1])
    # mask to clear diagonals which are positive pairs
    pos_masks = paddle.eye(batch_size).astype('bool') 

    batch_topk = min(batch_size, self.topk)
    if self.direction == 'i2t' or self.direction == 'bi':
      d1 = diagonal.expand_as(scores) # same collumn for im2s (negative sentence)
      # compare every diagonal score to scores in its collumn
      # caption retrieval
      cost_s = (margin + scores - d1).clip(min=0)
      cost_s[pos_masks] =  0 
      if self.max_violation:
        cost_s, _ = paddle.topk(cost_s, batch_topk, axis=1)
        cost_s = cost_s / batch_topk
        if average_batch:
          cost_s = cost_s / batch_size
      else:
        if average_batch:
          cost_s = cost_s / (batch_size * (batch_size - 1))
      cost_s = paddle.sum(cost_s)

    if self.direction == 't2i' or self.direction == 'bi':
      d2 = diagonal.t().expand_as(scores) # same row for s2im (negative image)
      # compare every diagonal score to scores in its row
      cost_im = (margin + scores - d2).clip(min=0)
      cost_im[pos_masks] = 0 
      if self.max_violation:
        cost_im, _ = paddle.topk(cost_im, batch_topk, axis=0)
        cost_im = cost_im / batch_topk
        if average_batch:
          cost_im = cost_im / batch_size
      else:
        if average_batch:
          cost_im = cost_im / (batch_size * (batch_size - 1))
      cost_im = paddle.sum(cost_im)

    if self.direction == 'i2t':
      return cost_s
    elif self.direction == 't2i':
      return cost_im
    else:
      return cost_s + cost_im
