import collections.abc

container_abcs = collections.abc
from ..registry import HEADS
from .base import BaseHead
from ..builder import build_loss


@HEADS.register()
class MoViNetHead(BaseHead):
    def __init__(self, loss_cfg=dict(name="CrossEntropyLoss")):
        super().__init__(loss_cfg=loss_cfg)

    def forward(self, x, *args):
        return x
