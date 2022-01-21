import paddle.nn as nn
# from reprod_log.utils import paddle2np

from EIVideo.paddlevideo.utils.manet_utils import fill_, zero_


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2D(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               dilation=dilation,
                               padding=dilation,
                               bias_attr=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2D(planes,
                               planes * 4,
                               kernel_size=1,
                               bias_attr=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    def __init__(self,
                 block,
                 layers,
                 output_stride,
                 BatchNorm,
                 pretrained=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2D(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias_attr=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block,
                                       64,
                                       layers[0],
                                       stride=strides[0],
                                       dilation=dilations[0],
                                       BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=strides[1],
                                       dilation=dilations[1],
                                       BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=strides[2],
                                       dilation=dilations[2],
                                       BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block,
                                         512,
                                         blocks=blocks,
                                         stride=strides[3],
                                         dilation=dilations[3],
                                         BatchNorm=BatchNorm)
        self.init_weight()



    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias_attr=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation, downsample,
                  BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      dilation=dilation,
                      BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self,
                      block,
                      planes,
                      blocks,
                      stride=1,
                      dilation=1,
                      BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias_attr=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  dilation=blocks[0] * dilation,
                  downsample=downsample,
                  BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(
                block(self.inplanes,
                      planes,
                      stride=1,
                      dilation=blocks[i] * dilation,
                      BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                fill_(m.weight, 1)
            elif isinstance(m, nn.BatchNorm2D):
                fill_(m.weight, 1)
                zero_(m.bias)
        return self.sublayers()




def ResNet101(output_stride, BatchNorm, pretrained=None):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   output_stride,
                   BatchNorm,
                   pretrained=pretrained)
    return model


def build_backbone(output_stride, BatchNorm, pretrained):
    return ResNet101(output_stride, BatchNorm, pretrained)


if __name__ == "__main__":
    import paddle

    model = ResNet101(BatchNorm=nn.BatchNorm2D,
                      pretrained=True,
                      output_stride=8)
    input = paddle.rand([1, 3, 512, 512])
    output, low_level_feat = model(input)
    print(output.shape)
    print(low_level_feat.shape)
    import json

    with open('output.txt', 'w') as f:
        json.dump(output.tolist(), f)
    with open('low_level_feat.txt', 'w') as f:
        json.dump(low_level_feat.tolist(), f)
