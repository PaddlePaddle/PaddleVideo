from .base import BaseHead
from ..registry import HEADS


@HEADS.register()
class EfficientGCNHead(BaseHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def forward(self, x):
        return x[0]
