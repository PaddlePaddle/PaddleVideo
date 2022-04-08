from paddlevideo.modeling.framework.recognizers.base import BaseRecognizer
import paddle
from ...registry import GENERATORS
from .ffa_metrics import psnr, compute_ssim
import numpy as np


@GENERATORS.register()
class FFANet(BaseRecognizer):
    """Example Recognizer model framework."""

    def forward_net(self, imgs):
        """model forward method"""
        preds = self.backbone(imgs)
        return preds

    def train_step(self, data_batch):
        """Define how the model is going to train, from input to output.
        """
        hazes = data_batch[0]
        clears = data_batch[1]
        preds = self.forward_net(hazes)
        loss_metrics = self.head.loss(clears, preds)
        # print(loss_metrics)
        outputs = {}
        outputs['loss'] = loss_metrics
        outputs['preds'] = preds
        return outputs  # loss_metrics

    def val_step(self, data_batch):
        hazes = data_batch[0]
        clears = data_batch[1]
        preds = self.forward_net(hazes)
        loss_metrics = self.head.loss(clears, preds)
        # print(loss_metrics)
        outputs = {}
        outputs['loss'] = loss_metrics
        outputs['preds'] = preds
        ssim_eval = 0
        for j in range(3):
            ssim_eval += compute_ssim(
                np.transpose(np.squeeze(preds.cpu().numpy(), 0),
                             (1, 2, 0))[:, :, j],
                np.transpose(np.squeeze(clears.cpu().numpy(), 0),
                             (1, 2, 0))[:, :, j]).item()
        ssim_eval /= 3
        psnr_eval = psnr(preds, clears)
        outputs['ssim'] = ssim_eval
        outputs['psnr'] = psnr_eval
        return outputs

    def test_step(self, data_batch):
        """Define how the model is going to test, from input to output.
        """
        hazes = data_batch[0]
        clears = data_batch[1]
        preds = self.forward_net(hazes)
        loss_metrics = self.head.loss(clears, preds)
        #print(loss_metrics)
        outputs = {}
        outputs['loss'] = loss_metrics
        outputs['preds'] = preds
        return outputs  #loss_metrics
