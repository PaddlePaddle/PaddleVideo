import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from EIVideo.paddlevideo.utils.manet_utils import kaiming_normal_


class _ASPPModule(nn.Layer):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2D(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias_attr=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU(True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                from EIVideo.paddlevideo.utils.manet_utils import fill_
                fill_(m.weight, 1)
                from EIVideo.paddlevideo.utils.manet_utils import zero_
                zero_(m.bias)


class ASPP(nn.Layer):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes,
                                 256,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 256,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 256,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 256,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.Conv2D(inplanes, 256, 1, stride=1, bias_attr=False),
            BatchNorm(256), nn.ReLU())
        self.conv1 = nn.Conv2D(1280, 256, 1, bias_attr=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.1)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.shape[2:],
                           mode='bilinear',
                           align_corners=True)
        x = paddle.concat((x1, x2, x3, x4, x5), axis=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x
        return self.dropout(x)

    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                # n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                # m.weight.normal_(0, math.sqrt(2. / n))
                kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                from EIVideo.paddlevideo.utils.manet_utils import fill_
                fill_(m.weight, 1)
                from EIVideo.paddlevideo.utils.manet_utils import zero_
                zero_(m.bias)


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)
