import paddle
import paddle.nn as nn
import numpy as np
from ..registry import ROI_EXTRACTORS
from .roi_extractor import RoIAlign
chaj_debug = 0
#
#from mmaction.utils import import_module_error_class
#
#try:
#    from mmcv.ops import RoIAlign, RoIPool
#except (ImportError, ModuleNotFoundError):
#
#    @import_module_error_class('mmcv-full')
#    class RoIAlign(nn.Module):
#        pass
#
#    @import_module_error_class('mmcv-full')
#    class RoIPool(nn.Module):
#        pass
#
#
#try:
#    from mmdet.models import ROI_EXTRACTORS
#    mmdet_imported = True
#except (ImportError, ModuleNotFoundError):
#    mmdet_imported = False

@ROI_EXTRACTORS.register()
class SingleRoIExtractor3D(nn.Layer):
    """Extract RoI features from a single level feature map.

    Args:
        roi_layer_type (str): Specify the RoI layer type. Default: 'RoIAlign'.
        featmap_stride (int): Strides of input feature maps. Default: 16.
        output_size (int | tuple): Size or (Height, Width). Default: 16.
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
            Default: 0.
        pool_mode (str, 'avg' or 'max'): pooling mode in each bin.
            Default: 'avg'.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
            Default: True.
        with_temporal_pool (bool): if True, avgpool the temporal dim.
            Default: True.
        with_global (bool): if True, concatenate the RoI feature with global
            feature. Default: False.

    Note that sampling_ratio, pool_mode, aligned only apply when roi_layer_type
    is set as RoIAlign.
    """

    def __init__(self,
                 roi_layer_type='RoIAlign',
                 featmap_stride=16,
                 output_size=16,
                 sampling_ratio=0,
                 pool_mode='avg',
                 aligned=True,
                 with_temporal_pool=True,
                 with_global=False):
        super().__init__()
        self.roi_layer_type = roi_layer_type
        assert self.roi_layer_type in ['RoIPool', 'RoIAlign']
        self.featmap_stride = featmap_stride
        self.spatial_scale = 1. / self.featmap_stride

        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.pool_mode = pool_mode
        self.aligned = aligned

        self.with_temporal_pool = with_temporal_pool
        self.with_global = with_global

        #chajchaj, SingleRoIExtractor3D roi_layer_type: RoIAlign with_global: False
        if chaj_debug:
            print("chajchaj, single_straight3d.py,  SingleRoIExtractor3D roi_layer_type:", self.roi_layer_type, "with_global:", self.with_global)
        #TODO
        #if self.roi_layer_type == 'RoIPool':
        #    self.roi_layer = RoIPool(self.output_size, self.spatial_scale)
        #else:
        self.roi_layer = RoIAlign(
            #'resolution': 7, 'sampling_ratio': 0, 'aligned': True, 'spatial_scale': [0.25, 0.125, 0.0625, 0.03125, 0.015625]
            resolution = self.output_size,
            spatial_scale = self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            #pool_mode=self.pool_mode,
            aligned=self.aligned)
        #self.global_pool = nn.AdaptiveAvgPool2d(self.output_size)

    def init_weights(self):
        pass

    # The shape of feat is N, C, T, H, W
    def forward(self, feat, rois, rois_num):
        #chajchaj, feat_type: <class 'tuple'> feat_len: 2 self.with_temporal_pool: True self.with_global: False
        if chaj_debug:
            print("chajchaj, single_straight3d, feat_type:", type(feat), "feat_len:", len(feat), "self.with_temporal_pool:",self.with_temporal_pool, "self.with_global:", self.with_global, "feat[0].shape:",feat[0].shape,"feat[1].shape:",feat[1].shape)
        #if not isinstance(feat, tuple):
        #    feat = (feat, )
        if len(feat) >= 2:
            assert self.with_temporal_pool
        if self.with_temporal_pool:
            xi = 0
            for x in feat:
                xi = xi + 1
                if chaj_debug:
                    print("chajchaj, single_straight3d, with_temporal_pool, xi:", xi, "x:",x)
                #y = torch.mean(x, 2, keepdim=True)
                y = paddle.mean(x, 2, keepdim=True)
                if chaj_debug:
                    print("chajchaj, single_straight3d, with_temporal_pool, x.shape:",x.shape)#,"y.shape:",y.shape)
                    print("chajchaj, single_straight3d, with_temporal_pool, y.shape:",y.shape)
                    print("chajchaj, single_straight3d, with_temporal_pool, y:",y)
            #feat = [torch.mean(x, 2, keepdim=True) for x in feat]
            feat = [paddle.mean(x, 2, keepdim=True) for x in feat]
        #feat = torch.cat(feat, axis=1)
        feat = paddle.concat(feat, axis=1) # merge slow and fast
        if chaj_debug:
            print("chajchaj, single_straight3d, af cat feat.shape:", feat.shape, "feat:", feat)
            print("chajchaj, single_straight3d, rois_num:",rois_num, "rois:",rois)
        roi_feats = []
        #for t in range(feat.size(2)):
        for t in range(feat.shape[2]):
            data_index = np.array([t]).astype('int32')
            index = paddle.to_tensor(data_index)
            frame_feat = paddle.index_select(feat, index, axis=2)
            if chaj_debug:
                print("chajchaj, single_straight3d, t:", t, "bf squeeze, frame_feat.shape:", frame_feat.shape, "frame_feat:",frame_feat)
            #frame_feat = paddle.squeeze(frame_feat)
            frame_feat = paddle.squeeze(frame_feat, 
                                        axis=2) #避免N=1时, 第一维度被删除.
            if chaj_debug:
                print("chajchaj, single_straight3d, t:", t, "af squeeze, frame_feat.shape:", frame_feat.shape, "frame_feat:",frame_feat)
            #frame_feat = feat[:, :, t].contiguous()

            roi_feat = self.roi_layer(frame_feat, rois, rois_num)
            if chaj_debug:
                print("chajchaj, single_straight3d, t:", t, "roi_feat.shape:", roi_feat.shape, "frame_feat.shape:",frame_feat.shape) #, "rois.shape:", rois.shape)
                print("chajchaj, single_straight3d, t:", t, "self.roi_layer:", self.roi_layer, "roi_feat:", roi_feat, "frame_feat:",frame_feat)
            #if self.with_global:
            #    global_feat = self.global_pool(frame_feat.contiguous())
            #    inds = rois[:, 0].type(torch.int64)
            #    global_feat = global_feat[inds]
            #    roi_feat = torch.cat([roi_feat, global_feat], dim=1)
            #    roi_feat = roi_feat.contiguous()
            roi_feats.append(roi_feat)

        #ret = torch.stack(roi_feats, dim=2)
        ret = paddle.stack(roi_feats, axis=2)
        if chaj_debug:
            print("chajchaj, single_straight3d, af stack ret.shape:", ret.shape, "ret:", ret)
        return ret


#if mmdet_imported:
#    ROI_EXTRACTORS.register_module()(SingleRoIExtractor3D)
