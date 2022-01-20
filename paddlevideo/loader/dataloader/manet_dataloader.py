from .base import BaseDataLoader
from ..registry import DATALOADERS


@DATALOADERS.register()
class ManetDataLoaderStage2(BaseDataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 collate_fn_cfg=None,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 **cfg):
        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False

        super().__init__(dataset,
                         feed_list=None,
                         return_list=return_list,
                         batch_sampler=batch_sampler,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         num_workers=num_workers,
                         use_buffer_reader=True,
                         use_shared_memory=False,
                         timeout=timeout,
                         worker_init_fn=worker_init_fn,
                         **cfg)
        if sampler is not None:
            self.batch_sampler.sampler = sampler
