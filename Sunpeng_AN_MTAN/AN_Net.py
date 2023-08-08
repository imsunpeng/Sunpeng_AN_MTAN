import torch
from torch import nn
import torch.nn.functional as F


class AttBlock(nn.Module):
    def __init__(self, ing_ch, inl_ch, out_ch):
        super(AttBlock, self).__init__()

        self.Wg = nn.Sequential(
            nn.Conv3d(ing_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(out_ch), )

        self.Wl = nn.Sequential(
            nn.Conv3d(inl_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(out_ch), )

        self.psi = nn.Sequential(
            nn.Conv3d(out_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        g1 = self.Wg(x1)
        x1 = self.Wl(x2)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x2 * psi


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class AN(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(AN, self).__init__()
        self.has_dropout = has_dropout

        #self.Maxpool = DownsamplingConvBlock(1, 1, normalization=normalization)
        self.Maxpool = DownsamplingConvBlock(1, 2 * n_filters, normalization=normalization)
        self.Maxpool1 = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)
        self.Maxpool2 = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)
        self.Maxpool3 = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_one = ConvBlock(2, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)
        self.attBlock1 = AttBlock(n_filters * 2, n_filters * 2, n_filters)

        self.block_two = ConvBlock(3, n_filters * 4, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)
        self.attBlock2 = AttBlock(n_filters * 4, n_filters * 4, n_filters* 2)

        self.block_three = ConvBlock(3, n_filters * 8, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)
        self.attBlock3 = AttBlock(n_filters * 8, n_filters * 8, n_filters * 4)

        self.block_four = ConvBlock(3, n_filters * 16, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)
        self.attBlock4 = AttBlock(n_filters * 16, n_filters * 16, n_filters * 8)

        self.block_five = ConvBlock(3, n_filters * 32, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(3, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x11 = self.Maxpool(input)
        x22 = self.Maxpool1(x11)
        x33 = self.Maxpool2(x22)
        x44 = self.Maxpool3(x33)
        # print(x11.shape)

        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)
        #print(x11.shape, x1_dw.shape )
        x01 = self.attBlock1(x1_dw, x11)
        #print(x04.shape)
        x1_dw = torch.cat((x01, x1_dw), dim=1)


        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)
        x02 = self.attBlock2(x2_dw, x22)
        x2_dw = torch.cat((x02, x2_dw), dim=1)


        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)
        x03 = self.attBlock3(x3_dw, x33)
        x3_dw = torch.cat((x03, x3_dw), dim=1)


        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)
        x04 = self.attBlock4(x4_dw, x44)
        x4_dw = torch.cat((x04, x4_dw), dim=1)


        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        # if self.has_dropout:
        #     x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        # if self.has_dropout:
        #     x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out

    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        self.featuremap_center = features[-1].detach()
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out



if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format

    model = AN(n_channels=1, n_classes=2)
    input = torch.randn(4, 1, 128, 128, 96)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
    print("AN have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))