import numpy as np
import paddle
import paddle.nn as nn

import paddle.nn.functional as F
from EIVideo.paddlevideo.utils.manet_utils import int_, float_, long_, load
from EIVideo.paddlevideo.utils.manet_utils import kaiming_normal_

#############################################################GLOBAL_DIST_MAP

MODEL_UNFOLD = True
WRONG_LABEL_PADDING_DISTANCE = 1e20


def _pairwise_distances(x, y, ys=None):
    """Computes pairwise squared l2 distances between tensors x and y.
    Args:
    x: Tensor of shape [n, feature_dim].
    y: Tensor of shape [m, feature_dim].
    Returns:
    Float32 distances tensor of shape [n, m].
    """

    xs = paddle.sum(x * x, 1)
    xs = xs.unsqueeze(1)
    if ys is None:
        ys = paddle.sum(y * y, 1)
        ys = ys.unsqueeze(0)
    else:
        ys = ys
    d = xs + ys - 2. * paddle.matmul(x, paddle.t(y))
    return d, ys


##################
def _flattened_pairwise_distances(reference_embeddings, query_embeddings, ys):
    """Calculates flattened tensor of pairwise distances between ref and query.
    Args:
    reference_embeddings: Tensor of shape [..., embedding_dim],
      the embedding vectors for the reference frame
    query_embeddings: Tensor of shape [n_query_images, height, width,
      embedding_dim], the embedding vectors for the query frames.
    Returns:
    A distance tensor of shape [reference_embeddings.size / embedding_dim,
    query_embeddings.size / embedding_dim]
    """
    embedding_dim = query_embeddings.shape[-1]
    reference_embeddings = reference_embeddings.reshape([-1, embedding_dim])
    first_dim = -1
    query_embeddings = query_embeddings.reshape([first_dim, embedding_dim])
    dists, ys = _pairwise_distances(query_embeddings, reference_embeddings, ys)
    return dists, ys


def _nn_features_per_object_for_chunk(reference_embeddings, query_embeddings,
                                      wrong_label_mask, k_nearest_neighbors,
                                      ys):
    """Extracts features for each object using nearest neighbor attention.
  Args:
    reference_embeddings: Tensor of shape [n_chunk, embedding_dim],
      the embedding vectors for the reference frame.
    query_embeddings: Tensor of shape [m_chunk, embedding_dim], the embedding
      vectors for the query frames.
    wrong_label_mask:
    k_nearest_neighbors: Integer, the number of nearest neighbors to use.
  Returns:
    nn_features: A float32 tensor of nearest neighbor features of shape
      [m_chunk, n_objects, feature_dim].
    """
    #    reference_embeddings_key = reference_embeddings
    #    query_embeddings_key = query_embeddings
    dists, ys = _flattened_pairwise_distances(reference_embeddings,
                                              query_embeddings, ys)

    dists = (paddle.unsqueeze(dists, 1) +
             paddle.unsqueeze(float_(wrong_label_mask), 0) *
             WRONG_LABEL_PADDING_DISTANCE)
    if k_nearest_neighbors == 1:
        features = paddle.min(dists, 2, keepdim=True)
    else:
        dists, _ = paddle.topk(-dists, k=k_nearest_neighbors, axis=2)
        dists = -dists
        valid_mask = (dists < WRONG_LABEL_PADDING_DISTANCE)
        masked_dists = dists * valid_mask.float()
        pad_dist = paddle.max(masked_dists, axis=2, keepdim=True)[0].tile(
            (1, 1, masked_dists.shape[-1]))
        dists = paddle.where(valid_mask, dists, pad_dist)
        # take mean of distances
        features = paddle.mean(dists, axis=2, keepdim=True)

    return features, ys


###
def _selected_pixel(ref_labels_flat, ref_emb_flat):
    index_list = paddle.arange(len(ref_labels_flat))
    index_list = index_list
    index_ = paddle.masked_select(index_list, ref_labels_flat != -1)

    index_ = long_(index_)
    ref_labels_flat = paddle.index_select(ref_labels_flat, index_, 0)
    ref_emb_flat = paddle.index_select(ref_emb_flat, index_, 0)

    return ref_labels_flat, ref_emb_flat


###


def _nearest_neighbor_features_per_object_in_chunks(reference_embeddings_flat,
                                                    query_embeddings_flat,
                                                    reference_labels_flat,
                                                    ref_obj_ids,
                                                    k_nearest_neighbors,
                                                    n_chunks, **cfg):
    """Calculates the nearest neighbor features per object in chunks to save mem.
    Uses chunking to bound the memory use.
    Args:
    reference_embeddings_flat: Tensor of shape [n, embedding_dim],
      the embedding vectors for the reference frame.
    query_embeddings_flat: Tensor of shape [m, embedding_dim], the embedding
      vectors for the query frames.
    reference_labels_flat: Tensor of shape [n], the class labels of the
      reference frame.
    ref_obj_ids: int tensor of unique object ids in the reference labels.
    k_nearest_neighbors: Integer, the number of nearest neighbors to use.
    n_chunks: Integer, the number of chunks to use to save memory
      (set to 1 for no chunking).
    Returns:
    nn_features: A float32 tensor of nearest neighbor features of shape
      [m, n_objects, feature_dim].
    """

    # reference_embeddings_flat = reference_embeddings_flat.cpu()
    # query_embeddings_flat = query_embeddings_flat.cpu()
    # reference_labels_flat = reference_labels_flat.cpu()
    # ref_obj_ids = ref_obj_ids.cpu()

    chunk_size = int_(
        np.ceil((float_(query_embeddings_flat.shape[0]) / n_chunks).numpy()))
    if cfg.get('test_mode'):
        reference_labels_flat, reference_embeddings_flat = _selected_pixel(
            reference_labels_flat, reference_embeddings_flat)
    wrong_label_mask = (reference_labels_flat != paddle.unsqueeze(
        ref_obj_ids, 1))
    all_features = []
    for n in range(n_chunks):
        if n == 0:
            ys = None
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_embeddings_flat_chunk = query_embeddings_flat[
                chunk_start:chunk_end]
        features, ys = _nn_features_per_object_for_chunk(
            reference_embeddings_flat, query_embeddings_flat_chunk,
            wrong_label_mask, k_nearest_neighbors, ys)
        all_features.append(features)
    if n_chunks == 1:
        nn_features = all_features[0]
    else:
        nn_features = paddle.concat(all_features, axis=0)
    return nn_features


def nearest_neighbor_features_per_object(reference_embeddings,
                                         query_embeddings,
                                         reference_labels,
                                         k_nearest_neighbors,
                                         gt_ids=None,
                                         n_chunks=100,
                                         **cfg):
    """Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
    reference_embeddings: Tensor of shape [height, width, embedding_dim],
      the embedding vectors for the reference frame.
    query_embeddings: Tensor of shape [n_query_images, height, width,
      embedding_dim], the embedding vectors for the query frames.
    reference_labels: Tensor of shape [height, width, 1], the class labels of
      the reference frame.
    max_neighbors_per_object: Integer, the maximum number of candidates
      for the nearest neighbor query per object after subsampling,
      or 0 for no subsampling.
    k_nearest_neighbors: Integer, the number of nearest neighbors to use.
    gt_ids: Int tensor of shape [n_objs] of the sorted unique ground truth
      ids in the first frame. If None, it will be derived from
      reference_labels.
    n_chunks: Integer, the number of chunks to use to save memory
      (set to 1 for no chunking).
    Returns:
    nn_features: A float32 tensor of nearest neighbor features of shape
      [n_query_images, height, width, n_objects, feature_dim].
    gt_ids: An int32 tensor of the unique sorted object ids present
      in the reference labels.
    """
    # reference_embeddings = reference_embeddings.detach().cpu()
    # query_embeddings = query_embeddings.detach().cpu()
    # reference_labels = reference_labels.detach().cpu()

    assert (reference_embeddings.shape[:2] == reference_labels.shape[:2])
    h, w, _ = query_embeddings.shape
    reference_labels_flat = reference_labels.reshape([-1])
    if gt_ids is None:
        ref_obj_ids = paddle.unique(reference_labels_flat)[-1]
        ref_obj_ids = np.arange(0, ref_obj_ids + 1)
        gt_ids = paddle.to_tensor(ref_obj_ids)
        gt_ids = int_(gt_ids)
    else:
        gt_ids = int_(paddle.arange(0, gt_ids + 1))

    embedding_dim = query_embeddings.shape[-1]
    query_embeddings_flat = query_embeddings.reshape([-1, embedding_dim])
    reference_embeddings_flat = reference_embeddings.reshape(
        [-1, embedding_dim])
    nn_features = _nearest_neighbor_features_per_object_in_chunks(
        reference_embeddings_flat, query_embeddings_flat,
        reference_labels_flat, gt_ids, k_nearest_neighbors, n_chunks, **cfg)
    nn_features_dim = nn_features.shape[-1]
    nn_features = nn_features.reshape(
        [1, h, w, gt_ids.shape[0], nn_features_dim])
    return nn_features.cuda(), gt_ids


########################################################################LOCAL_DIST_MAP


def local_pairwise_distances2(x, y, max_distance=9):
    """Computes pairwise squared l2 distances using a local search window.
    Naive implementation using map_fn.
    Used as a slow fallback for when correlation_cost is not available.
    Args:
    x: Float32 tensor of shape [height, width, feature_dim].
    y: Float32 tensor of shape [height, width, feature_dim].
    max_distance: Integer, the maximum distance in pixel coordinates
      per dimension which is considered to be in the search window.
    Returns:
    Float32 distances tensor of shape
      [height, width, (2 * max_distance + 1) ** 2].
    """
    ori_h, ori_w, _ = x.shape
    x = paddle.transpose(x, [2, 0, 1]).unsqueeze(0)
    x = F.avg_pool2d(x, (2, 2), (2, 2))
    y = paddle.transpose(y, [2, 0, 1]).unsqueeze(0)
    y = F.avg_pool2d(y, (2, 2), (2, 2))

    _, channels, height, width = x.shape
    padding_val = 1e20
    padded_y = F.pad(y,
                     (max_distance, max_distance, max_distance, max_distance),
                     mode='constant',
                     value=padding_val)
    offset_y = F.unfold(padded_y, kernel_sizes=[height, width]).reshape(
        [1, channels, height, width, -1])
    x = x.reshape([1, channels, height, width, 1])
    minus = x - offset_y
    dists = paddle.sum(paddle.multiply(minus, minus),
                       axis=1).reshape([1, height, width,
                                        -1]).transpose([0, 3, 1, 2])
    dists = (paddle.nn.functional.sigmoid(dists) - 0.5) * 2
    dists = F.interpolate(dists,
                          size=[ori_h, ori_w],
                          mode='bilinear',
                          align_corners=True)
    dists = dists.squeeze(0).transpose([1, 2, 0])
    return dists


def local_previous_frame_nearest_neighbor_features_per_object(
        prev_frame_embedding,
        query_embedding,
        prev_frame_labels,
        gt_ids,
        max_distance=12):
    """Computes nearest neighbor features while only allowing local matches.
  Args:
    prev_frame_embedding: Tensor of shape [height, width, embedding_dim],
      the embedding vectors for the last frame.
    query_embedding: Tensor of shape [height, width, embedding_dim],
      the embedding vectors for the query frames.
    prev_frame_labels: Tensor of shape [height, width, 1], the class labels of
      the previous frame.
    gt_ids: Int Tensor of shape [n_objs] of the sorted unique ground truth
      ids in the first frame.
    max_distance: Integer, the maximum distance allowed for local matching.
  Returns:
    nn_features: A float32 np.array of nearest neighbor features of shape
      [1, height, width, n_objects, 1].
    """
    #     print(query_embedding.shape, prev_frame_embedding.shape)
    #     print(query_embedding.place, prev_frame_embedding.place)
    #     query_embedding = query_embedding.cpu()
    #     prev_frame_embedding = prev_frame_embedding.cpu()
    #     prev_frame_labels = prev_frame_labels.cpu()
    #     print(prev_frame_labels.place, prev_frame_embedding.place, query_embedding.place)

    d = local_pairwise_distances2(query_embedding,
                                  prev_frame_embedding,
                                  max_distance=max_distance)
    height, width = prev_frame_embedding.shape[:2]

    if MODEL_UNFOLD:

        labels = float_(prev_frame_labels).transpose([2, 0, 1]).unsqueeze(0)
        padded_labels = F.pad(labels, (
            2 * max_distance,
            2 * max_distance,
            2 * max_distance,
            2 * max_distance,
        ))
        offset_labels = F.unfold(padded_labels,
                                 kernel_sizes=[height, width],
                                 strides=[2,
                                          2]).reshape([height, width, -1, 1])
        offset_masks = paddle.equal(
            offset_labels,
            float_(gt_ids).unsqueeze(0).unsqueeze(0).unsqueeze(0))
    else:

        masks = paddle.equal(prev_frame_labels,
                             gt_ids.unsqueeze(0).unsqueeze(0))
        padded_masks = nn.functional.pad(masks, (
            0,
            0,
            max_distance,
            max_distance,
            max_distance,
            max_distance,
        ))
        offset_masks = []
        for y_start in range(2 * max_distance + 1):
            y_end = y_start + height
            masks_slice = padded_masks[y_start:y_end]
            for x_start in range(2 * max_distance + 1):
                x_end = x_start + width
                offset_mask = masks_slice[:, x_start:x_end]
                offset_masks.append(offset_mask)
        offset_masks = paddle.stack(offset_masks, axis=2)

    d_tiled = d.unsqueeze(-1).tile((1, 1, 1, gt_ids.shape[0]))
    pad = paddle.ones_like(d_tiled)
    d_masked = paddle.where(offset_masks, d_tiled, pad)
    dists = paddle.min(d_masked, axis=2)
    dists = dists.reshape([1, height, width, gt_ids.shape[0], 1])

    return dists


##############################################################


#################
class _res_block(nn.Layer):
    def __init__(self, in_dim, out_dim, **cfg):
        super(_res_block, self).__init__()
        self.conv1 = nn.Conv2D(in_dim,
                               out_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = paddle.nn.BatchNorm2D(out_dim, momentum=cfg['train_bn_mom'])
        self.conv2 = nn.Conv2D(out_dim,
                               out_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = paddle.nn.BatchNorm2D(out_dim, momentum=cfg['train_bn_mom'])

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x += res
        return x


####################
class IntSegHead(nn.Layer):
    def __init__(self, in_dim, emb_dim, **cfg):
        super(IntSegHead, self).__init__()
        self.conv1 = nn.Conv2D(in_dim,
                               emb_dim,
                               kernel_size=7,
                               stride=1,
                               padding=3)
        self.bn1 = paddle.nn.BatchNorm2D(emb_dim, momentum=cfg['train_bn_mom'])
        self.relu1 = nn.ReLU(True)
        self.res1 = _res_block(emb_dim, emb_dim, **cfg)
        self.res2 = _res_block(emb_dim, emb_dim, **cfg)
        self.conv2 = nn.Conv2D(256,
                               emb_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = paddle.nn.BatchNorm2D(emb_dim, momentum=cfg['train_bn_mom'])
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.Conv2D(emb_dim, 1, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class _split_separable_conv2d(nn.Layer):
    def __init__(self, in_dim, out_dim, kernel_size=7, **cfg):
        super(_split_separable_conv2d, self).__init__()
        self.conv1 = nn.Conv2D(in_dim,
                               in_dim,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=int((kernel_size - 1) / 2),
                               groups=in_dim)
        self.relu1 = nn.ReLU(True)
        self.bn1 = paddle.nn.BatchNorm2D(in_dim, momentum=cfg['train_bn_mom'])
        self.conv2 = nn.Conv2D(in_dim, out_dim, kernel_size=1, stride=1)
        self.relu2 = nn.ReLU(True)
        self.bn2 = paddle.nn.BatchNorm2D(out_dim, momentum=cfg['train_bn_mom'])
        kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class DynamicSegHead(nn.Layer):
    def __init__(self, in_dim, embed_dim, **cfg):
        super(DynamicSegHead, self).__init__()
        self.layer1 = _split_separable_conv2d(in_dim, embed_dim, **cfg)
        self.layer2 = _split_separable_conv2d(embed_dim, embed_dim, **cfg)
        self.layer3 = _split_separable_conv2d(embed_dim, embed_dim, **cfg)
        self.layer4 = _split_separable_conv2d(embed_dim, embed_dim, **cfg)
        self.conv = nn.Conv2D(embed_dim, 1, 1, 1)
        kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv(x)
        return x


from ..registry import HEADS
"""
覆盖原理
class c1:
    def __init__(self):
        self.a = 1


class c2(c1):
    def __init__(self):
        super(c2, self).__init__()
        self.a = 2


c = c2()
print(c.a)

"""


@HEADS.register()
class IntVOS(nn.Layer):
    def __init__(self, feature_extracter, **cfg):
        super(IntVOS, self).__init__()
        self.feature_extracter = feature_extracter  ##embedding extractor
        self.feature_extracter.cls_conv = nn.Sequential()
        self.feature_extracter.upsample4 = nn.Sequential()
        self.semantic_embedding = None
        self.seperate_conv = nn.Conv2D(cfg['model_aspp_outdim'],
                                       cfg['model_aspp_outdim'],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       groups=cfg['model_aspp_outdim'])
        self.bn1 = paddle.nn.BatchNorm2D(cfg['model_aspp_outdim'],
                                         momentum=cfg['train_bn_mom'])
        self.relu1 = nn.ReLU(True)
        self.embedding_conv = nn.Conv2D(cfg['model_aspp_outdim'],
                                        cfg['model_semantic_embedding_dim'], 1,
                                        1)
        self.relu2 = nn.ReLU(True)
        self.bn2 = paddle.nn.BatchNorm2D(cfg['model_semantic_embedding_dim'],
                                         momentum=cfg['train_bn_mom'])
        self.semantic_embedding = nn.Sequential(*[
            self.seperate_conv, self.bn1, self.relu1, self.embedding_conv,
            self.bn2, self.relu2
        ])

        for m in self.semantic_embedding:
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.dynamic_seghead = DynamicSegHead(
            in_dim=cfg['model_semantic_embedding_dim'] + 3,
            embed_dim=cfg['model_head_embedding_dim'],
            **cfg)  # propagation segm head
        if cfg['model_useintseg']:
            self.inter_seghead = IntSegHead(
                in_dim=cfg['model_semantic_embedding_dim'] + 3,
                emb_dim=cfg['model_head_embedding_dim'],
                **cfg)
        else:
            self.inter_seghead = DynamicSegHead(
                in_dim=cfg['model_semantic_embedding_dim'] + 2,
                embed_dim=cfg['model_head_embedding_dim'],
                **cfg)  # interaction segm head
        self.pretrained = cfg.get('pretrained', None)
        self.cfg = cfg

    def init_weights(self):
        if isinstance(self.pretrained, str) and self.pretrained.strip() != "":
            self.set_state_dict(load(self.pretrained, self.state_dict()))
            print('loaded pretrained model')

    def loss(self, **kwargs):
        return self.loss_func(**kwargs)

    def forward(self,
                x=None,
                ref_scribble_label=None,
                previous_frame_mask=None,
                normalize_nearest_neighbor_distances=True,
                use_local_map=True,
                seq_names=None,
                gt_ids=None,
                k_nearest_neighbors=1,
                global_map_tmp_dic=None,
                local_map_dics=None,
                interaction_num=None,
                start_annotated_frame=None,
                frame_num=None):

        x = self.extract_feature(x)
        #         print('extract_feature:', x.mean().item())
        ref_frame_embedding, previous_frame_embedding, current_frame_embedding = paddle.split(
            x, num_or_sections=3, axis=0)

        if global_map_tmp_dic is None:
            dic = self.prop_seghead(
                ref_frame_embedding,
                previous_frame_embedding,
                current_frame_embedding,
                ref_scribble_label,
                previous_frame_mask,
                normalize_nearest_neighbor_distances,
                use_local_map,
                seq_names,
                gt_ids,
                k_nearest_neighbors,
                global_map_tmp_dic,
                local_map_dics,
                interaction_num,
                start_annotated_frame,
                frame_num,
                self.dynamic_seghead,
            )
            return dic

        else:
            dic, global_map_tmp_dic = self.prop_seghead(
                ref_frame_embedding,
                previous_frame_embedding,
                current_frame_embedding,
                ref_scribble_label,
                previous_frame_mask,
                normalize_nearest_neighbor_distances,
                use_local_map,
                seq_names,
                gt_ids,
                k_nearest_neighbors,
                global_map_tmp_dic,
                local_map_dics,
                interaction_num,
                start_annotated_frame,
                frame_num,
                self.dynamic_seghead,
            )
            return dic, global_map_tmp_dic

    def extract_feature(self, x):
        x = self.feature_extracter(x)
        x = self.semantic_embedding(x)
        return x

    def prop_seghead(
        self,
        ref_frame_embedding=None,
        previous_frame_embedding=None,
        current_frame_embedding=None,
        ref_scribble_label=None,
        previous_frame_mask=None,
        normalize_nearest_neighbor_distances=True,
        use_local_map=True,
        seq_names=None,
        gt_ids=None,
        k_nearest_neighbors=1,
        global_map_tmp_dic=None,
        local_map_dics=None,
        interaction_num=None,
        start_annotated_frame=None,
        frame_num=None,
        dynamic_seghead=None,
    ):
        """return: feature_embedding,global_match_map,local_match_map,previous_frame_mask"""
        ###############
        cfg = self.cfg
        global_map_tmp_dic = global_map_tmp_dic
        dic_tmp = {}
        bs, c, h, w = current_frame_embedding.shape
        if cfg.get('test_mode'):
            scale_ref_scribble_label = float_(ref_scribble_label)
        else:
            scale_ref_scribble_label = paddle.nn.functional.interpolate(
                float_(ref_scribble_label), size=(h, w), mode='nearest')
        scale_ref_scribble_label = int_(scale_ref_scribble_label)
        scale_previous_frame_label = paddle.nn.functional.interpolate(
            float_(previous_frame_mask), size=(h, w), mode='nearest')
        scale_previous_frame_label = int_(scale_previous_frame_label)
        for n in range(bs):
            seq_current_frame_embedding = current_frame_embedding[n]
            seq_ref_frame_embedding = ref_frame_embedding[n]
            seq_prev_frame_embedding = previous_frame_embedding[n]
            seq_ref_frame_embedding = seq_ref_frame_embedding.transpose(
                [1, 2, 0])
            seq_current_frame_embedding = seq_current_frame_embedding.transpose(
                [1, 2, 0])
            seq_ref_scribble_label = scale_ref_scribble_label[n].transpose(
                [1, 2, 0])
            #########Global Map
            nn_features_n, ref_obj_ids = nearest_neighbor_features_per_object(
                reference_embeddings=seq_ref_frame_embedding,
                query_embeddings=seq_current_frame_embedding,
                reference_labels=seq_ref_scribble_label,
                k_nearest_neighbors=k_nearest_neighbors,
                gt_ids=gt_ids[n],
                n_chunks=10)
            if normalize_nearest_neighbor_distances:
                nn_features_n = (paddle.nn.functional.sigmoid(nn_features_n) -
                                 0.5) * 2

            #             print(nn_features_n)

            ###
            if global_map_tmp_dic is not None:  ###when testing, use global map memory
                if seq_names[n] not in global_map_tmp_dic:
                    global_map_tmp_dic[seq_names[n]] = paddle.ones_like(
                        nn_features_n).tile([1000, 1, 1, 1, 1])
                nn_features_n = paddle.where(
                    nn_features_n <= global_map_tmp_dic[seq_names[n]][
                        frame_num[n]].unsqueeze(0), nn_features_n,
                    global_map_tmp_dic[seq_names[n]][frame_num[n]].unsqueeze(
                        0))

                #                 print('detach 1')
                #                 print(nn_features_n.shape)
                # nn_features_n = nn_features_n.detach()
                global_map_tmp_dic[seq_names[n]][
                    frame_num[n]] = nn_features_n.detach()[0]

            #########################Local dist map
            seq_prev_frame_embedding = seq_prev_frame_embedding.transpose(
                [1, 2, 0])
            seq_previous_frame_label = scale_previous_frame_label[n].transpose(
                [1, 2, 0])

            if use_local_map:
                prev_frame_nn_features_n = local_previous_frame_nearest_neighbor_features_per_object(
                    prev_frame_embedding=seq_prev_frame_embedding,
                    query_embedding=seq_current_frame_embedding,
                    prev_frame_labels=seq_previous_frame_label,
                    gt_ids=ref_obj_ids,
                    max_distance=cfg['model_max_local_distance'])
            else:
                prev_frame_nn_features_n, _ = nearest_neighbor_features_per_object(
                    reference_embeddings=seq_prev_frame_embedding,
                    query_embeddings=seq_current_frame_embedding,
                    reference_labels=seq_previous_frame_label,
                    k_nearest_neighbors=k_nearest_neighbors,
                    gt_ids=gt_ids[n],
                    n_chunks=20)
                prev_frame_nn_features_n = (
                    paddle.nn.functional.sigmoid(prev_frame_nn_features_n) -
                    0.5) * 2

            #             print(prev_frame_nn_features_n.mean().item(), prev_frame_nn_features_n.shape, interaction_num)  # o
            #############
            if local_map_dics is not None:  ##When testing, use local map memory
                local_map_tmp_dic, local_map_dist_dic = local_map_dics
                if seq_names[n] not in local_map_dist_dic:
                    print(seq_names[n], 'not in local_map_dist_dic')
                    local_map_dist_dic[seq_names[n]] = paddle.zeros(1000, 9)
                if seq_names[n] not in local_map_tmp_dic:
                    print(seq_names[n], 'not in local_map_tmp_dic')
                    local_map_tmp_dic[seq_names[n]] = paddle.zeros_like(
                        prev_frame_nn_features_n).unsqueeze(0).tile(
                            [1000, 9, 1, 1, 1, 1])
                #                 print(local_map_dist_dic[seq_names[n]].shape)
                #                 print('detach 2')
                # prev_frame_nn_features_n = prev_frame_nn_features_n.detach()
                local_map_dist_dic[seq_names[n]][
                    frame_num[n], interaction_num -
                    1] = 1.0 / (abs(frame_num[n] - start_annotated_frame)
                                )  # bugs fixed.
                local_map_tmp_dic[seq_names[n]][
                    frame_num[n],
                    interaction_num - 1] = prev_frame_nn_features_n.squeeze(
                        0).detach()  # bugs fixed.
                if interaction_num == 1:
                    prev_frame_nn_features_n = local_map_tmp_dic[seq_names[n]][
                        frame_num[n]][interaction_num - 1]
                    prev_frame_nn_features_n = prev_frame_nn_features_n.unsqueeze(
                        0)
                else:
                    if local_map_dist_dic[seq_names[n]][frame_num[n]][interaction_num - 1] > \
                            local_map_dist_dic[seq_names[n]][frame_num[n]][interaction_num - 2]:
                        prev_frame_nn_features_n = local_map_tmp_dic[
                            seq_names[n]][frame_num[n]][interaction_num - 1]
                        prev_frame_nn_features_n = prev_frame_nn_features_n.unsqueeze(
                            0)
                    else:
                        prev_frame_nn_features_n = local_map_tmp_dic[
                            seq_names[n]][frame_num[n]][interaction_num - 2]
                        prev_frame_nn_features_n = prev_frame_nn_features_n.unsqueeze(
                            0)

                local_map_dics = (local_map_tmp_dic, local_map_dist_dic)

            to_cat_previous_frame = (
                float_(seq_previous_frame_label) == float_(ref_obj_ids)
            )  # float comparision?

            to_cat_current_frame_embedding = current_frame_embedding[
                n].unsqueeze(0).tile((ref_obj_ids.shape[0], 1, 1, 1))

            to_cat_nn_feature_n = nn_features_n.squeeze(0).transpose(
                [2, 3, 0, 1])
            to_cat_previous_frame = float_(
                to_cat_previous_frame.unsqueeze(-1).transpose([2, 3, 0, 1]))
            to_cat_prev_frame_nn_feature_n = prev_frame_nn_features_n.squeeze(
                0).transpose([2, 3, 0, 1])
            to_cat = paddle.concat(
                (to_cat_current_frame_embedding, to_cat_nn_feature_n,
                 to_cat_prev_frame_nn_feature_n, to_cat_previous_frame), 1)
            pred_ = dynamic_seghead(to_cat)
            pred_ = pred_.transpose([1, 0, 2, 3])
            dic_tmp[seq_names[n]] = pred_

        if global_map_tmp_dic is None:
            return dic_tmp
        else:
            if local_map_dics is None:
                return dic_tmp, global_map_tmp_dic
            else:
                return dic_tmp, global_map_tmp_dic, local_map_dics

    def int_seghead(self,
                    ref_frame_embedding=None,
                    ref_scribble_label=None,
                    prev_round_label=None,
                    normalize_nearest_neighbor_distances=True,
                    global_map_tmp_dic=None,
                    local_map_dics=None,
                    interaction_num=None,
                    seq_names=None,
                    gt_ids=None,
                    k_nearest_neighbors=1,
                    frame_num=None,
                    first_inter=True):
        dic_tmp = {}
        bs, c, h, w = ref_frame_embedding.shape
        scale_ref_scribble_label = paddle.nn.functional.interpolate(
            float_(ref_scribble_label), size=(h, w), mode='nearest')
        scale_ref_scribble_label = int_(scale_ref_scribble_label)
        if not first_inter:
            scale_prev_round_label = paddle.nn.functional.interpolate(
                float_(prev_round_label), size=(h, w), mode='nearest')
            scale_prev_round_label = int_(scale_prev_round_label)
        n_chunks = 500
        for n in range(bs):

            gt_id = paddle.arange(0, gt_ids[n] + 1)

            gt_id = int_(gt_id)

            seq_ref_frame_embedding = ref_frame_embedding[n]

            ########################Local dist map
            seq_ref_frame_embedding = paddle.transpose(seq_ref_frame_embedding,
                                                       [1, 2, 0])
            seq_ref_scribble_label = paddle.transpose(
                scale_ref_scribble_label[n], [1, 2, 0])
            nn_features_n = local_previous_frame_nearest_neighbor_features_per_object(
                prev_frame_embedding=seq_ref_frame_embedding,
                query_embedding=seq_ref_frame_embedding,
                prev_frame_labels=seq_ref_scribble_label,
                gt_ids=gt_id,
                max_distance=self.cfg['model_max_local_distance'])

            #######
            ######################Global map update
            if seq_names[n] not in global_map_tmp_dic:
                global_map_tmp_dic[seq_names[n]] = paddle.ones_like(
                    nn_features_n).tile([1000, 1, 1, 1, 1])
            nn_features_n_ = paddle.where(
                nn_features_n <=
                global_map_tmp_dic[seq_names[n]][frame_num[n]].unsqueeze(0),
                nn_features_n,
                global_map_tmp_dic[seq_names[n]][frame_num[n]].unsqueeze(0))

            ###

            ###
            #             print('detach 3')
            # nn_features_n_ = nn_features_n_.detach()
            global_map_tmp_dic[seq_names[n]][
                frame_num[n]] = nn_features_n_.detach()[0]
            ##################Local map update
            if local_map_dics is not None:
                local_map_tmp_dic, local_map_dist_dic = local_map_dics
                if seq_names[n] not in local_map_dist_dic:
                    local_map_dist_dic[seq_names[n]] = paddle.zeros([1000, 9])
                if seq_names[n] not in local_map_tmp_dic:
                    local_map_tmp_dic[seq_names[n]] = paddle.ones_like(
                        nn_features_n).unsqueeze(0).tile([1000, 9, 1, 1, 1, 1])
                local_map_dist_dic[seq_names[n]][frame_num[n]][interaction_num
                                                               - 1] = 0

                local_map_dics = (local_map_tmp_dic, local_map_dist_dic)

            ##################
            to_cat_current_frame_embedding = ref_frame_embedding[n].unsqueeze(
                0).tile((gt_id.shape[0], 1, 1, 1))
            to_cat_nn_feature_n = nn_features_n.squeeze(0).transpose(
                [2, 3, 0, 1])

            to_cat_scribble_mask_to_cat = (
                float_(seq_ref_scribble_label) == float_(gt_id)
            )  # float comparision?
            to_cat_scribble_mask_to_cat = float_(
                to_cat_scribble_mask_to_cat.unsqueeze(-1).transpose(
                    [2, 3, 0, 1]))
            if not first_inter:
                seq_prev_round_label = scale_prev_round_label[n].transpose(
                    [1, 2, 0])

                to_cat_prev_round_to_cat = (
                    float_(seq_prev_round_label) == float_(gt_id)
                )  # float comparision?
                to_cat_prev_round_to_cat = float_(
                    to_cat_prev_round_to_cat.unsqueeze(-1).transpose(
                        [2, 3, 0, 1]))
            else:
                to_cat_prev_round_to_cat = paddle.zeros_like(
                    to_cat_scribble_mask_to_cat)
                to_cat_prev_round_to_cat[0] = 1.

            to_cat = paddle.concat(
                (to_cat_current_frame_embedding, to_cat_scribble_mask_to_cat,
                 to_cat_prev_round_to_cat), 1)

            pred_ = self.inter_seghead(to_cat)
            pred_ = pred_.transpose([1, 0, 2, 3])
            dic_tmp[seq_names[n]] = pred_
        if local_map_dics is None:
            return dic_tmp
        else:
            return dic_tmp, local_map_dics
