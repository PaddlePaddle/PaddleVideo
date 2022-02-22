import collections.abc

container_abcs = collections.abc
from ..registry import HEADS
from .base import BaseHead
from ..builder import build_loss


@HEADS.register()
class MoViNetHead(BaseHead):

    def __init__(
        self,
        #  num_classes,
        #  in_channels,
        # loss_cfg=dict(name="CrossEntropyLoss")
    ):
        super().__init__()
        #super().__init__(num_classes, in_channels, loss_cfg)
        # self.num_classes = num_classes
        # self.in_channels = in_channels
        #self.loss_func = build_loss(loss_cfg)

    def forward(self, x):
        return x
