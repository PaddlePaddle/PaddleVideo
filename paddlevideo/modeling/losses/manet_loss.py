import paddle
import paddle.nn as nn
from ..registry import LOSSES
from .base import BaseWeightedLoss
from ...utils.manet_utils import float_, long_


@LOSSES.register()
class Topk_CrossEntropyLoss(BaseWeightedLoss):
    def __init__(self,
                 top_k_percent_pixels=None,
                 hard_example_mining_step=100000):
        super(Topk_CrossEntropyLoss, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        if top_k_percent_pixels is not None:
            assert (top_k_percent_pixels > 0 and top_k_percent_pixels < 1)
        self.hard_example_mining_step = hard_example_mining_step
        if self.top_k_percent_pixels == None:
            self.celoss = nn.CrossEntropyLoss(ignore_index=255,
                                              reduction='mean')
        else:
            self.celoss = nn.CrossEntropyLoss(ignore_index=255,
                                              reduction='none')

    def forward(self, dic_tmp, label_dic, step, **kwargs):

        final_loss = 0
        for seq_name in dic_tmp.keys():
            pred_logits = dic_tmp[seq_name]
            gts = long_(label_dic[seq_name])
            if self.top_k_percent_pixels == None:
                final_loss += self.celoss(pred_logits, gts)
            else:
                # Only compute the loss for top k percent pixels.
                # First, compute the loss for all pixels. Note we do not put the loss
                # to loss_collection and set reduction = None to keep the shape.
                num_pixels = float(pred_logits.shape[2] * pred_logits.shape[3])
                pred_logits = pred_logits.reshape([
                    pred_logits.shape[1],
                    pred_logits.shape[2] * pred_logits.shape[3]
                ]).transpose([1, 0])
                gts = gts.reshape([gts.shape[1] * gts.shape[2]])
                pixel_losses = self.celoss(pred_logits, gts).reshape([1, -1])
                if self.hard_example_mining_step == 0:
                    top_k_pixels = int(self.top_k_percent_pixels * num_pixels)
                else:
                    ratio = min(1.0,
                                step / float(self.hard_example_mining_step))
                    top_k_pixels = int((ratio * self.top_k_percent_pixels +
                                        (1.0 - ratio)) * num_pixels)
                top_k_loss, top_k_indices = paddle.topk(pixel_losses,
                                                        k=top_k_pixels,
                                                        axis=1)

                final_loss += paddle.mean(top_k_loss)
        return final_loss
