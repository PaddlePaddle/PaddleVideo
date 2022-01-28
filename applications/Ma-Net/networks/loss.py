import paddle
import paddle.nn as nn
import os


class Added_BCEWithLogitsLoss(nn.Layer):
    def __init__(self,
                 top_k_percent_pixels=None,
                 hard_example_mining_step=100000):
        super(Added_BCEWithLogitsLoss, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        if top_k_percent_pixels is not None:
            assert (top_k_percent_pixels > 0 and top_k_percent_pixels < 1)
        self.hard_example_mining_step = hard_example_mining_step
        if self.top_k_percent_pixels == None:
            self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.bceloss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, dic_tmp, y, step):
        final_loss = 0
        for seq_name in dic_tmp.keys():
            pred_logits = dic_tmp[seq_name]
            gts = y[seq_name]
            if self.top_k_percent_pixels == None:
                final_loss += self.bceloss(pred_logits, gts)
            else:
                # Only compute the loss for top k percent pixels.
                # First, compute the loss for all pixels. Note we do not put the loss
                # to loss_collection and set reduction = None to keep the shape.
                num_pixels = float(pred_logits.shape[2] * pred_logits.shape[3])
                pred_logits = pred_logits.view(
                    -1, pred_logits.shape[1],
                    pred_logits.shape[2] * pred_logits.shape[3])
                gts = gts.view(-1, gts.shape[1], gts.shape[2] * gts.shape[3])
                pixel_losses = self.bceloss(pred_logits, gts)
                if self.hard_example_mining_step == 0:
                    top_k_pixels = int(self.top_k_percent_pixels * num_pixels)
                else:
                    ratio = min(1.0,
                                step / float(self.hard_example_mining_step))
                    top_k_pixels = int((ratio * self.top_k_percent_pixels +
                                        (1.0 - ratio)) * num_pixels)
                _, top_k_indices = paddle.topk(pixel_losses,
                                               k=top_k_pixels,
                                               axis=2)

                final_loss += nn.BCEWithLogitsLoss(weight=top_k_indices,
                                                   reduction='mean')(
                                                       pred_logits, gts)
        return final_loss


class Added_CrossEntropyLoss(nn.Layer):
    def __init__(self,
                 top_k_percent_pixels=None,
                 hard_example_mining_step=100000):
        super(Added_CrossEntropyLoss, self).__init__()
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

    def forward(self, dic_tmp, y, step):
        final_loss = 0
        for seq_name in dic_tmp.keys():
            pred_logits = dic_tmp[seq_name]
            gts = y[seq_name]
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


class AddedEdge_CrossEntropyLoss(nn.Layer):
    def __init__(self,
                 top_k_percent_pixels=None,
                 hard_example_mining_step=100000):
        super(AddedEdge_CrossEntropyLoss, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        if top_k_percent_pixels is not None:
            assert (top_k_percent_pixels > 0 and top_k_percent_pixels < 1)
        self.hard_example_mining_step = hard_example_mining_step
        self.celoss = None

    def forward(self, pred_logits, gts, step):
        pos_num = paddle.sum(gts == 1, dtype='float32')
        neg_num = paddle.sum(gts == 0, dtype='float32')

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = paddle.to_tensor([weight_neg, weight_pos])
        if self.top_k_percent_pixels == None:
            sig_pred_logits = paddle.nn.functional.sigmoid(pred_logits)
            self.bceloss = nn.BCEWithLogitsLoss(pos_weight=weight_pos,
                                                reduction='mean')
            if paddle.sum(gts) == 0:
                dcloss = 0
            else:
                dcloss = (paddle.sum(sig_pred_logits * sig_pred_logits) +
                          paddle.sum(gts * gts)) / (
                              paddle.sum(2 * sig_pred_logits * gts) + 1e-5)
            final_loss = 0.1 * self.bceloss(pred_logits, gts) + dcloss
        else:
            self.celoss = nn.CrossEntropyLoss(weight=weights,
                                              ignore_index=255,
                                              reduction='none')
            num_pixels = float(pred_logits.shape[2] * pred_logits.shape[3])
            pred_logits = pred_logits.view(
                -1, pred_logits.shape[1],
                pred_logits.shape[2] * pred_logits.shape[3])
            gts = gts.view(-1, gts.shape[2] * gts.shape[3])
            pixel_losses = self.celoss(pred_logits, gts)
            if self.hard_example_mining_step == 0:
                top_k_pixels = int(self.top_k_percent_pixels * num_pixels)
            else:
                ratio = min(1.0, step / float(self.hard_example_mining_step))
                top_k_pixels = int((ratio * self.top_k_percent_pixels +
                                    (1.0 - ratio)) * num_pixels)
            top_k_loss, top_k_indices = paddle.topk(pixel_losses,
                                                    k=top_k_pixels,
                                                    axis=1)

            final_loss = paddle.mean(top_k_loss)
        return final_loss
