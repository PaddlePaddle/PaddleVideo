import cv2
import numpy as np
import paddle
from paddlevideo.utils import get_logger

from .base import BaseMetric
from .registry import METRIC

logger = get_logger("paddlevideo")


@METRIC.register
class RMSEMetric(BaseMetric):
    def __init__(self, data_size, batch_size, log_interval=1, scale=4):
        """prepare for rmse and ssim metrics.
        """
        super().__init__(data_size, batch_size, log_interval)
        self.rmses = []
        self.ssims = []
        self.scale = scale

    def update(self, batch_id: int, data: paddle.Tensor,
               outputs: paddle.Tensor):
        """update metrics during each iter
        """
        labels = data[1]
        height, width = outputs.shape[-2:]
        shave_border = self.scale

        # shave border
        outputs = outputs[:, :, shave_border:height - shave_border,
                          shave_border:width - shave_border]
        labels = labels[:, :, shave_border:height - shave_border,
                        shave_border:width - shave_border]

        # mask
        mask = (outputs != 0).astype('float32')
        labels = mask * labels

        # compute rmse & ssim
        rmse = self._compute_rmse(outputs, labels)
        ssim = self._compute_ssim(outputs, labels)

        # NOTE(shipping): deal with multi cards validate
        if self.world_size > 1:
            rmse = paddle.distributed.all_reduce(
                rmse, op=paddle.distributed.ReduceOp.SUM) / self.world_size

        self.rmses.append(rmse.numpy())
        self.ssims.append(ssim.numpy())
        print(f"rmse = {self.rmses[-1]} ssim = {self.ssims[-1]}")
        # preds ensemble
        if batch_id % self.log_interval == 0:
            logger.info("[TEST] Processing batch {}/{} ...".format(
                batch_id,
                self.data_size // (self.batch_size * self.world_size)))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        logger.info('[TEST] finished, avg_rmse = {}, avg_ssim = {}.'.format(
            np.mean(np.array(self.rmses)), np.mean(np.array(self.ssims))))

    def _compute_rmse(self, imgs1: paddle.Tensor,
                      imgs2: paddle.Tensor) -> paddle.Tensor:

        imdff = imgs1 * 255.0 - imgs2 * 255.0
        rmse = paddle.sqrt(paddle.mean(imdff**2))
        return rmse

    def _compute_ssim(self, imgs1: paddle.Tensor,
                      imgs2: paddle.Tensor) -> paddle.Tensor:
        imgs1 = imgs1 * 255.0
        imgs2 = imgs2 * 255.0

        # convert to numpy for ssim computation
        imgs1 = imgs1.squeeze().numpy()  # [h,w]
        imgs2 = imgs2.squeeze().numpy()  # [h,w]
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        imgs1 = imgs1.astype(np.float64)
        imgs2 = imgs2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(imgs1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(imgs2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(imgs1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(imgs2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(imgs1 * imgs2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))
        ssim_map = ssim_map.mean()

        # convert back to tensor when gathering among multi gpus.
        ssim_map_tensor = paddle.to_tensor(ssim_map)
        return ssim_map_tensor
