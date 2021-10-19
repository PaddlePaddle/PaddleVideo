import paddle
import numpy as np

#from ..builder import BBOX_SAMPLERS
#from .base_sampler import BaseSampler
from ..registry import BBOX_SAMPLERS

chaj_align = 0 #1
chaj_debug = 0


#class SamplingResult(util_mixins.NiceRepr):
class SamplingResult():
    """Bbox sampling result.  """

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        if chaj_debug:
            print("chajchaj, SamplingResult, pos_inds:",pos_inds,"neg_inds:",neg_inds,"assign_result:",assign_result,"assign_result.gt_inds:",assign_result.gt_inds, "bboxes:",bboxes,"gt_bboxes:",gt_bboxes,"gt_flags:",gt_flags)
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        #self.pos_bboxes = bboxes[pos_inds]
        #self.neg_bboxes = bboxes[neg_inds]
        #self.pos_is_gt = gt_flags[pos_inds]
        self.pos_bboxes = paddle.index_select(bboxes,pos_inds)
        self.neg_bboxes = paddle.index_select(bboxes,neg_inds)
        self.pos_is_gt  = paddle.index_select(gt_flags,pos_inds)
        #print("chajchaj, SamplingResult, self.pos_bboxes:",self.pos_bboxes, "self.neg_bboxes:",self.neg_bboxes, "self.pos_is_gt:",self.pos_is_gt) 
        self.num_gts = gt_bboxes.shape[0]
        #self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_assigned_gt_inds = paddle.index_select(assign_result.gt_inds,pos_inds) - 1
        if chaj_debug:
            print("chajchaj, SamplingResult, self.pos_assigned_gt_inds:", self.pos_assigned_gt_inds, "gt_bboxes.numel():",gt_bboxes.numel().numpy())


        #if gt_bboxes.numel() == 0:
        if gt_bboxes.numel().numpy()[0] == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if chaj_debug:
                print("chajchaj, SamplingResult, len(gt_bboxes.shape):",len(gt_bboxes.shape))
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            #self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
            self.pos_gt_bboxes = paddle.index_select(gt_bboxes, self.pos_assigned_gt_inds)

        if assign_result.labels is not None:
            #self.pos_gt_labels = assign_result.labels[pos_inds]
            self.pos_gt_labels = paddle.index_select(assign_result.labels, pos_inds)
        else:
            self.pos_gt_labels = None
        #print("chajchaj, SamplingResult, self.pos_bboxes:",self.pos_bboxes,"self.neg_bboxes:",self.neg_bboxes)
        if chaj_debug:
            print("chajchaj, SamplingResult, self.pos_is_gt:",self.pos_is_gt,"self.num_gts:",self.num_gts,"self.pos_assigned_gt_inds:",self.pos_assigned_gt_inds)
            print("chajchaj, SamplingResult, self.pos_gt_bboxes:", self.pos_gt_bboxes, "self.pos_gt_labels:",self.pos_gt_labels.numpy())

    @property
    def bboxes(self):
        """torch.Tensor: concatenated positive and negative boxes"""
        #ret = torch.cat([self.pos_bboxes, self.neg_bboxes])
        ret = paddle.concat([self.pos_bboxes, self.neg_bboxes])
        if chaj_debug:
            print("chajchaj, sampling_result.py, bboxes, ret:", ret)
        return ret



#@BBOX_SAMPLERS.register_module()
@BBOX_SAMPLERS.register()
#class RandomSampler(BaseSampler):
class RandomSampler():
    """Random sampler.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int, optional): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool, optional): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
 
        #from mmdet.core.bbox import demodata
        #super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
        #                                    add_gt_as_proposals)
        #self.rng = demodata.ensure_rng(kwargs.get('rng', None))
    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        if chaj_debug:
            print("chajchaj,  BaseSampler, self.num:",self.num, "self.pos_fraction", self.pos_fraction, "self.add_gt_as_proposals:",self.add_gt_as_proposals)
            print("chajchaj,  BaseSampler,  assign_result:",assign_result,"bboxes:", bboxes, "gt_bboxes:", gt_bboxes, "gt_labels:",gt_labels)

        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]
        if chaj_debug:
            print("chajchaj,  BaseSampler, af bboxes, bboxes:", bboxes)

        #gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        gt_flags = paddle.full([bboxes.shape[0], ], 0, dtype='int32')
        if chaj_debug:
            print("chajchaj,  BaseSampler, gt_flags:",gt_flags)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            #bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            bboxes = paddle.concat([gt_bboxes, bboxes])
            if chaj_debug:
                print("chajchaj,  BaseSampler, af  concat, bboxes:",bboxes)
                assign_result.cj_print("bf add_gt_")
            assign_result.add_gt_(gt_labels)
            if chaj_debug:
                assign_result.cj_print("af add_gt_")
            #gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_ones = paddle.full([gt_bboxes.shape[0], ], 1, dtype='int32')
            #gt_flags = torch.cat([gt_ones, gt_flags])
            gt_flags = paddle.concat([gt_ones, gt_flags])
            if chaj_debug:
                print("chajchaj,  BaseSampler, af add_gt_as_proposals, bboxes:", bboxes)
                print("chajchaj,  BaseSampler, af add_gt_as_proposals, gt_flags:",gt_flags)

        #1. 得到正样本的数量, inds
        num_expected_pos = int(self.num * self.pos_fraction)
        #pos_inds = self.pos_sampler._sample_pos(
        pos_inds = self._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        #pos_inds = pos_inds.unique()
        if chaj_debug:
            print("chajchaj,  BaseSampler, af _sample_pos, pos_inds:",pos_inds)
        #TODO:下面这一行代码在idx=196时报错, 待提卡片, 先用np解决
        #pos_inds = paddle.unique(pos_inds)
        pos_inds = paddle.to_tensor(np.unique(pos_inds.numpy()))
        if chaj_debug:
            print("chajchaj,  BaseSampler, af unique, pos_inds:",pos_inds)

        #2. 得到负样本的数量, inds
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if chaj_debug:
            print("chajchaj,  BaseSampler, af numel, num_expected_neg:",num_expected_neg, "self.neg_pos_ub:",self.neg_pos_ub)
        #if self.neg_pos_ub >= 0:
        #    _pos = max(1, num_sampled_pos)
        #    neg_upper_bound = int(self.neg_pos_ub * _pos)
        #    if num_expected_neg > neg_upper_bound:
        #        num_expected_neg = neg_upper_bound
        #neg_inds = self.neg_sampler._sample_neg(
        neg_inds = self._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        #print("chajchaj,  BaseSampler, af _sample_neg, neg_inds:",neg_inds)
        if chaj_debug:
            print("chajchaj,  BaseSampler, af _sample_neg, neg_inds:",neg_inds.numel())
        #neg_inds = neg_inds.unique()
        #neg_inds = paddle.unique(neg_inds)
        neg_inds = paddle.to_tensor(np.unique(neg_inds.numpy()))
        #print("chajchaj,  BaseSampler, af unique, neg_inds:",neg_inds)
        if chaj_debug:
            print("chajchaj,  BaseSampler, af unique, neg_inds:",neg_inds.numel())

        #3. 得到sampling result
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        if chaj_debug:
            print("chajchaj, BaseSampler, sampling_result:",sampling_result)
        return sampling_result
    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        #is_tensor = isinstance(gallery, torch.Tensor)
        #if not is_tensor:
        #    if torch.cuda.is_available():
        #        device = torch.cuda.current_device()
        #    else:
        #        device = 'cpu'
        #    gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        #perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        perm = paddle.arange(gallery.numel())[:num]
        if not chaj_align:
            perm = paddle.randperm(gallery.numel())[:num]
        if chaj_debug:
            print("chajchaj, random_sampler.py, random_choice, perm:",perm, "chaj_align:",chaj_align)
        #rand_inds = gallery[perm]
        rand_inds = paddle.index_select(gallery, perm)
        if chaj_debug:
            print("chajchaj, random_sampler.py, random_choice, af perm, rand_inds:",rand_inds)
        #if not is_tensor:
        #    rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        #1.首先看一下给的bboxes里面有哪些label是大于0的 得到了他们的index
        #pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        pos_inds = paddle.nonzero(assign_result.gt_inds, as_tuple=False)#.squeeze()
        if chaj_debug:
            print("chajchaj,random_sampler.py, _sample_pos, assign_result.gt_inds:", assign_result.gt_inds) 
            print("chajchaj,random_sampler.py, _sample_pos, pos_inds:",pos_inds,"pos_inds.numel():",pos_inds.numel()) 

        #2. 只要这个pos_inds的数目不是0个 这些就都可以是positive sample
        # 当pos_inds的数目小于num_expected(想要的sample的最大数目), 就直接用这个pos_inds
        # 反之就从这么多index里随机采样num_expected个出来
        #if pos_inds.numel() != 0:
        if pos_inds.numel().numpy()[0] != 0:
            pos_inds = pos_inds.squeeze() #(1)
        if chaj_debug:
            print("chajchaj,random_sampler.py, _sample_pos, af squeeze, pos_inds:",pos_inds) 
        #num_expected = 5 #TODO delete it
        #if pos_inds.numel() <= num_expected:
        if pos_inds.numel().numpy()[0] <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        if chaj_debug:
            print("chajchaj,random_sampler.py, _sample_neg, assign_result.gt_inds:",assign_result.gt_inds) 
        #neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        neg_inds = paddle.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        #print("chajchaj,random_sampler.py, _sample_neg, neg_inds:",neg_inds,"num_expected:",num_expected) 
        if chaj_debug:
            print("chajchaj,random_sampler.py, _sample_neg, neg_inds:",neg_inds.numel(),"num_expected:",num_expected) 
        #TODO:下面这一行代码在idx=196时报错, 待提卡片, 先用np解决
        #if neg_inds.numel() != 0:
        if neg_inds.numel().numpy()[0] != 0:
            neg_inds = neg_inds.squeeze() #(1)
        #print("chajchaj,random_sampler.py, _sample_neg, af squeeze, neg_inds:", neg_inds) 
        if chaj_debug:
            print("chajchaj,random_sampler.py, _sample_neg, af squeeze, neg_inds:", neg_inds.numel()) 
        #num_expected = 2 #TODO delete it
        #TODO:下面这一行代码在idx=196时报错, 待提卡片, 先用np解决
        #if len(neg_inds) <= num_expected:
        if (neg_inds.numel().numpy()[0]) <= num_expected.numpy()[0]:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)
