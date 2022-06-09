from abc import abstractmethod
from ... import builder
import paddle
import paddle.nn as nn


class BaseRecognizer(nn.Layer):
    """Base class for recognizers.

    All recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``train_step``, supporting to forward when training.
    - Methods:``valid_step``, supporting to forward when validating.
    - Methods:``test_step``, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        head (dict): Classification head to process feature.

    """
    def __init__(self, backbone=None, head=None, runtime_cfg=None):

        super().__init__()
        if backbone is not None:
            self.backbone = builder.build_backbone(backbone)
            if hasattr(self.backbone, 'init_weights'):
                self.backbone.init_weights()
        else:
            self.backbone = None
        if head is not None:
            self.head_name = head.name
            self.head = builder.build_head(head)
            if hasattr(self.head, 'init_weights'):
                self.head.init_weights()
        else:
            self.head = None

        # Settings when the model is running,
        # such as 'avg_type'
        self.runtime_cfg = runtime_cfg

    def forward(self, data_batch, mode='infer', **kwargs):
        """
        1. Define how the model is going to run, from input to output.
        2. Console of train, valid, test or infer step
        3. Set mode='infer' is used for saving inference model, refer to tools/export_model.py
        """
        if mode == 'train':
            return self.train_step(data_batch, **kwargs)
        elif mode == 'valid':
            return self.val_step(data_batch, **kwargs)
        elif mode == 'test':
            return self.test_step(data_batch, **kwargs)
        elif mode == 'infer':
            return self.infer_step(data_batch, **kwargs)
        else:
            raise NotImplementedError

    @abstractmethod
    def train_step(self, data_batch, **kwargs):
        """Training step.
        """
        raise NotImplementedError

    @abstractmethod
    def val_step(self, data_batch, **kwargs):
        """Validating step.
        """
        raise NotImplementedError

    @abstractmethod
    def test_step(self, data_batch, **kwargs):
        """Test step.
        """
        raise NotImplementedError

    @abstractmethod
    def infer_step(self, data_batch, **kwargs):
        """Infer step.
        """
        raise NotImplementedError

    def _gather_from_gpu(self,
                         gather_object: paddle.Tensor,
                         concat_axis=0) -> paddle.Tensor:
        """gather Tensor from all gpus into a list and concatenate them on `concat_axis`.
        Args:
            gather_object (paddle.Tensor): gather object Tensor
            concat_axis (int, optional): axis for concatenation. Defaults to 0.
        Returns:
            paddle.Tensor: gatherd & concatenated Tensor
        """
        gather_object_list = []
        paddle.distributed.all_gather(gather_object_list, gather_object)
        return paddle.concat(gather_object_list, axis=concat_axis)
