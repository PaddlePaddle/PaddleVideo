import paddle

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register()
class TVLoss(BaseWeightedLoss):
    def __init__(self):
        super(TVLoss, self).__init__()

    def _forward(self, x: paddle.Tensor) -> paddle.Tensor:
        h_x = x.shape[2]
        w_x = x.shape[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = paddle.pow((x[:, :, 1:, :]) - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = paddle.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return h_tv / count_h + w_tv / count_w

    def _tensor_size(self, t: paddle.Tensor) -> int:
        return t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3]
