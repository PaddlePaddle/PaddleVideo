import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from EIVideo.paddlevideo.utils.manet_utils import kaiming_normal_


class Decoder(nn.Layer):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn' or backbone == 'resnet_edge':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2D(low_level_inplanes, 48, 1, bias_attr=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU(True)
        self.last_conv = nn.Sequential(
            nn.Conv2D(304,
                      256,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias_attr=False), BatchNorm(256), nn.ReLU(True),
            nn.Sequential(),
            nn.Conv2D(256,
                      256,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias_attr=False), BatchNorm(256), nn.ReLU(True),
            nn.Sequential())
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x,
                          size=low_level_feat.shape[2:],
                          mode='bilinear',
                          align_corners=True)
        x = paddle.concat((x, low_level_feat), axis=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                from EIVideo.paddlevideo.utils.manet_utils import fill_
                fill_(m.weight, 1)
                from EIVideo.paddlevideo.utils.manet_utils import zero_
                zero_(m.bias)


def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)
