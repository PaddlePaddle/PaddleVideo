import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..registry import HEADS
from .base import BaseHead


class LossNetwork(paddle.nn.Layer):

    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name in range(len(self.vgg_layers)):
            x = self.vgg_layers[name](x)
            if str(name) in self.layer_name_mapping:
                output[self.layer_name_mapping[str(name)]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss) / len(loss)


@HEADS.register()
class FFANetHead(BaseHead):
    """ FFANet Head """

    def __init__(self, perloss=True):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.perloss = perloss
        if self.perloss:
            vgg_model = paddle.vision.models.vgg16(pretrained=False)
            # 加载pytorch-VGG16-pretrained的初始化参数
            path_state_dict = 'data/FFA/vgg16_pretrained_weight.pdparams'
            load_state_dict = paddle.load(path_state_dict)
            # net.load_dict(load_state_dict)
            vgg_model.load_dict({
                k.replace('module.', ''): v
                for k, v in load_state_dict.items()
            })
            self.loss2 = LossNetwork(vgg_model.features[:16])

    def loss(self, x1, x2):
        """model forward """
        #print('x1.shape={}'.format(x1.shape))
        loss = self.l1_loss(x1, x2)
        if self.perloss:
            loss += 0.04 * self.loss2(x1, x2)
        return loss
