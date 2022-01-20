from __future__ import absolute_import
from collections import defaultdict
import numpy as np
from .base import BaseSampler
from paddlevideo.loader.registry import SAMPLERS


@SAMPLERS.register()
class RandomIdentitySampler(BaseSampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, dataset, num_instances=1):
        super(RandomIdentitySampler, self).__init__(data_source=dataset.info)
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, tmp_dic in enumerate(self.data_source):
            pid = tmp_dic['seq_name']
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = np.random.permutation(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances
