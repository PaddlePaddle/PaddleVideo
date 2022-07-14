import collections.abc

container_abcs = collections.abc
from ..registry import HEADS
from .base import BaseHead
from ..builder import build_loss


@HEADS.register()
class MoViNetHead(BaseHead):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args):
        return x
