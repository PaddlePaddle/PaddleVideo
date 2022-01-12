import paddle
import paddle.nn as nn

from ..registry import BACKBONES
from EIVideo.paddlevideo.modeling.backbones.aspp_manet import build_aspp
from EIVideo.paddlevideo.modeling.backbones.decoder_manet import build_decoder
from EIVideo.paddlevideo.modeling.backbones.resnet_manet import build_backbone


class FrozenBatchNorm2d(nn.Layer):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", paddle.ones(n))
        self.register_buffer("bias", paddle.zeros(n))
        self.register_buffer("running_mean", paddle.zeros(n))
        self.register_buffer("running_var", paddle.ones(n))

    def forward(self, x):
        if x.dtype == paddle.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


@BACKBONES.register()
class DeepLab(nn.Layer):
    def __init__(self,
                 backbone='resnet',
                 output_stride=16,
                 num_classes=21,
                 freeze_bn=False,
                 pretrained=None):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        if freeze_bn == True:
            print("Use frozen BN in DeepLab")
            BatchNorm = FrozenBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2D

        self.backbone = build_backbone(output_stride, BatchNorm, pretrained)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)


    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        return x

    def freeze_bn(self):
        for m in self.sublayers():
            if isinstance(m, nn.BatchNorm2D):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2D) or isinstance(
                        m[1], nn.BatchNorm2D):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2D) or isinstance(
                        m[1], nn.BatchNorm2D):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16)
    model.eval()
    input = paddle.rand([2, 3, 513, 513])
    output = model(input)
    print(output.shape)
