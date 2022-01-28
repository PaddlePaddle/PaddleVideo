import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Layer):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 BatchNorm=None):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2D(inplanes,
                               inplanes,
                               kernel_size,
                               stride,
                               0,
                               dilation,
                               groups=inplanes,
                               bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2D(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x,
                          self.conv1._kernel_size[0],
                          dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Layer):
    def __init__(self,
                 inplanes,
                 planes,
                 reps,
                 stride=1,
                 dilation=1,
                 BatchNorm=None,
                 start_with_relu=True,
                 grow_first=True,
                 is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2D(inplanes,
                                  planes,
                                  1,
                                  stride=stride,
                                  bias_attr=False)
            self.skipbn = BatchNorm(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU()
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(inplanes,
                                planes,
                                3,
                                1,
                                dilation,
                                BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(filters,
                                filters,
                                3,
                                1,
                                dilation,
                                BatchNorm=BatchNorm))
            rep.append(BatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(inplanes,
                                planes,
                                3,
                                1,
                                dilation,
                                BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(planes, planes, 3, 2, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(planes, planes, 3, 1, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x


class AlignedXception(nn.Layer):
    """
    Modified Alighed Xception
    """
    def __init__(self, output_stride, BatchNorm, pretrained=True):
        super(AlignedXception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2D(3, 32, 3, stride=2, padding=1, bias_attr=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2D(32, 64, 3, stride=1, padding=1, bias_attr=False)
        self.bn2 = BatchNorm(64)

        self.block1 = Block(64,
                            128,
                            reps=2,
                            stride=2,
                            BatchNorm=BatchNorm,
                            start_with_relu=False)
        self.block2 = Block(128,
                            256,
                            reps=2,
                            stride=2,
                            BatchNorm=BatchNorm,
                            start_with_relu=False,
                            grow_first=True)
        self.block3 = Block(256,
                            728,
                            reps=2,
                            stride=entry_block3_stride,
                            BatchNorm=BatchNorm,
                            start_with_relu=True,
                            grow_first=True,
                            is_last=True)

        # Middle flow
        self.block4 = Block(728,
                            728,
                            reps=3,
                            stride=1,
                            dilation=middle_block_dilation,
                            BatchNorm=BatchNorm,
                            start_with_relu=True,
                            grow_first=True)
        self.block5 = Block(728,
                            728,
                            reps=3,
                            stride=1,
                            dilation=middle_block_dilation,
                            BatchNorm=BatchNorm,
                            start_with_relu=True,
                            grow_first=True)
        self.block6 = Block(728,
                            728,
                            reps=3,
                            stride=1,
                            dilation=middle_block_dilation,
                            BatchNorm=BatchNorm,
                            start_with_relu=True,
                            grow_first=True)
        self.block7 = Block(728,
                            728,
                            reps=3,
                            stride=1,
                            dilation=middle_block_dilation,
                            BatchNorm=BatchNorm,
                            start_with_relu=True,
                            grow_first=True)
        self.block8 = Block(728,
                            728,
                            reps=3,
                            stride=1,
                            dilation=middle_block_dilation,
                            BatchNorm=BatchNorm,
                            start_with_relu=True,
                            grow_first=True)
        self.block9 = Block(728,
                            728,
                            reps=3,
                            stride=1,
                            dilation=middle_block_dilation,
                            BatchNorm=BatchNorm,
                            start_with_relu=True,
                            grow_first=True)
        self.block10 = Block(728,
                             728,
                             reps=3,
                             stride=1,
                             dilation=middle_block_dilation,
                             BatchNorm=BatchNorm,
                             start_with_relu=True,
                             grow_first=True)
        self.block11 = Block(728,
                             728,
                             reps=3,
                             stride=1,
                             dilation=middle_block_dilation,
                             BatchNorm=BatchNorm,
                             start_with_relu=True,
                             grow_first=True)
        self.block12 = Block(728,
                             728,
                             reps=3,
                             stride=1,
                             dilation=middle_block_dilation,
                             BatchNorm=BatchNorm,
                             start_with_relu=True,
                             grow_first=True)
        self.block13 = Block(728,
                             728,
                             reps=3,
                             stride=1,
                             dilation=middle_block_dilation,
                             BatchNorm=BatchNorm,
                             start_with_relu=True,
                             grow_first=True)
        self.block14 = Block(728,
                             728,
                             reps=3,
                             stride=1,
                             dilation=middle_block_dilation,
                             BatchNorm=BatchNorm,
                             start_with_relu=True,
                             grow_first=True)
        self.block15 = Block(728,
                             728,
                             reps=3,
                             stride=1,
                             dilation=middle_block_dilation,
                             BatchNorm=BatchNorm,
                             start_with_relu=True,
                             grow_first=True)
        self.block16 = Block(728,
                             728,
                             reps=3,
                             stride=1,
                             dilation=middle_block_dilation,
                             BatchNorm=BatchNorm,
                             start_with_relu=True,
                             grow_first=True)
        self.block17 = Block(728,
                             728,
                             reps=3,
                             stride=1,
                             dilation=middle_block_dilation,
                             BatchNorm=BatchNorm,
                             start_with_relu=True,
                             grow_first=True)
        self.block18 = Block(728,
                             728,
                             reps=3,
                             stride=1,
                             dilation=middle_block_dilation,
                             BatchNorm=BatchNorm,
                             start_with_relu=True,
                             grow_first=True)
        self.block19 = Block(728,
                             728,
                             reps=3,
                             stride=1,
                             dilation=middle_block_dilation,
                             BatchNorm=BatchNorm,
                             start_with_relu=True,
                             grow_first=True)

        # Exit flow
        self.block20 = Block(728,
                             1024,
                             reps=2,
                             stride=1,
                             dilation=exit_block_dilations[0],
                             BatchNorm=BatchNorm,
                             start_with_relu=True,
                             grow_first=False,
                             is_last=True)

        self.conv3 = SeparableConv2d(1024,
                                     1536,
                                     3,
                                     stride=1,
                                     dilation=exit_block_dilations[1],
                                     BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(1536)

        self.conv4 = SeparableConv2d(1536,
                                     1536,
                                     3,
                                     stride=1,
                                     dilation=exit_block_dilations[1],
                                     BatchNorm=BatchNorm)
        self.bn4 = BatchNorm(1536)

        self.conv5 = SeparableConv2d(1536,
                                     2048,
                                     3,
                                     stride=1,
                                     dilation=exit_block_dilations[1],
                                     BatchNorm=BatchNorm)
        self.bn5 = BatchNorm(2048)

        # Init weights
        self._init_weight()

        # Load pretrained model
        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                m.weight.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2D):
                from utils.api import fill_
                fill_(m.weight, 1)
                from utils.api import zero_
                zero_(m.bias)

    def _load_pretrained_model(self):
        import paddlehub as hub
        pretrain_dict = hub.Module(name="xception71_imagenet")
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in model_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    model_dict[k] = v
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.set_state_dict(state_dict)
