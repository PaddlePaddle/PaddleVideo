import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss


def bi_loss(scores, anchors, bgm_match_threshold=0.5):
    """
    cross_entropy loss
    :param scores: gt
    :param anchors: predict result
    :param bgm_match_threshold: threshold for selecting positive samples
    :return:
    """
    scores = paddle.reshape(scores, [scores.shape[-1]])
    anchors = paddle.reshape(anchors, [anchors.shape[-1]])
    # pmask = (scores> bgm_match_threshold).float()
    pmask = paddle.cast((scores > bgm_match_threshold), 'float32')
    num_positive = paddle.sum(pmask)
    num_entries = len(scores)
    ratio = num_entries / num_positive

    coef_0 = 0.5 * (ratio) / (ratio - 1)
    coef_1 = coef_0 * (ratio - 1)
    loss = coef_1 * pmask * paddle.log(anchors + 0.00001) + \
        coef_0 * (1.0 - pmask) * paddle.log(1.0 - anchors * 0.999999)
    loss = -1 * paddle.mean(loss)
    num_sample = [paddle.sum(pmask), ratio]
    return loss, num_sample


def BGM_loss_calc(anchors, match_scores):
    """BGM_loss_calc
    """
    loss_start_small, num_sample_start_small = bi_loss(match_scores, anchors)
    loss_dict = {"loss": loss_start_small, "num_sample": num_sample_start_small}
    return loss_dict


@LOSSES.register()
class BcnBgmLoss(BaseWeightedLoss):
    """BcnBgmLoss"""

    def forward(self, label, outputs):
        """Forward function.
        """
        loss_dict = BGM_loss_calc(outputs, label)
        return loss_dict["loss"]


def MultiplyList(myList):
    """multiplyList
    """
    result = 1
    for x in myList:
        result = result * x
    return [result]


@LOSSES.register()
class BcnModelLoss(BaseWeightedLoss):
    """BcnModelLoss"""

    def __init__(self):
        super().__init__()
        self.maskCE = nn.CrossEntropyLoss(
            ignore_index=-100, reduction='none')  # for cascade stages
        self.mse = nn.MSELoss(reduction='none')
        self.nll = nn.NLLLoss(ignore_index=-100,
                              reduction='none')  # for fusion stage

    def forward(self, predictions, adjust_weight, batch_target, mask):
        """Forward function.
        """
        loss = 0.
        num_stages = len(predictions) - 1
        balance_weight = [1.0] * num_stages
        batch_target = paddle.reshape(batch_target,
                                      MultiplyList(batch_target.shape))

        # num_stages is number of cascade stages
        for num_stage in range(num_stages):
            adjust_weight[num_stage].stop_gradient = True

            # balance_weight = a / b
            a = paddle.sum(adjust_weight[0], 2)
            a = paddle.reshape(a, MultiplyList(a.shape))
            a = paddle.cast(a, 'float32')

            b = paddle.sum(adjust_weight[num_stage], 2)
            b = paddle.reshape(b, MultiplyList(b.shape))
            b = paddle.cast(b, 'float32')

            balance_weight[num_stage] = paddle.mean(a / b)

            # calculate mask ce_loss
            p = predictions[num_stage]
            ce_p = paddle.transpose(p, [0, 2, 1])
            ce_p = paddle.reshape(
                ce_p, [ce_p.shape[0] * ce_p.shape[1], ce_p.shape[2]])
            ce_mask = adjust_weight[num_stage]
            ce_mask = paddle.reshape(ce_mask, MultiplyList(ce_mask.shape))

            ce_loss = 1 * balance_weight[num_stage] * paddle.mean(
                self.maskCE(ce_p, batch_target) * ce_mask)
            loss += ce_loss

            # calculate tmse
            loss += 0.3 * paddle.mean(
                paddle.clip(self.mse(
                    F.log_softmax(p[:, :, 1:], axis=1),
                    F.log_softmax(p.detach()[:, :, :-1], axis=1)),
                            min=0,
                            max=8) * mask[:, :, 1:])

        # fusion stage
        p = predictions[-1]
        nll_p = paddle.transpose(p, [0, 2, 1])
        nll_p = paddle.reshape(
            nll_p, [nll_p.shape[0] * nll_p.shape[1], nll_p.shape[2]])
        loss += paddle.mean(self.nll(paddle.log(nll_p), batch_target))

        loss += 0.5 * paddle.mean(
            paddle.clip(self.mse(F.log_softmax(p[:, :, 1:], axis=1),
                                 F.log_softmax(p.detach()[:, :, :-1], axis=1)),
                        min=0,
                        max=8) * mask[:, :, 1:])

        return loss
