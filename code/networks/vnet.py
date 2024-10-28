import torch
from torch import nn
import torch.nn.functional as F


class DepthWiseConv3d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv3d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv3d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


def Conv(in_channel,out_channel,kernel_size,**kwargs):
    return nn.Sequential(nn.Conv3d(in_channel,out_channel,kernel_size,**kwargs),
                         nn.BatchNorm3d(out_channel),
                         nn.LeakyReLU())

class Fconv(nn.Module):
    def __init__(self,n,in_channel,out_channel_list,middle_channel_list):
        super(Fconv, self).__init__()
        if n == 1:
            self.branch1_1=Conv(in_channel=in_channel,out_channel=middle_channel_list[0],kernel_size=1,stride=1,padding=0)
            self.branch1_2=Conv(in_channel=middle_channel_list[0],out_channel=out_channel_list[0],kernel_size=3,stride=1,padding=1)
            self.branch2_1=Conv(in_channel=in_channel,out_channel=middle_channel_list[1],kernel_size=1,stride=1,padding=0)
            self.branch2_2=Conv(in_channel=middle_channel_list[1],out_channel=middle_channel_list[2],kernel_size=(1,3,3),stride=1,padding=(0,1,1))
            self.branch2_3=Conv(in_channel=middle_channel_list[2],out_channel=middle_channel_list[3],kernel_size=(3,1,3),stride=1,padding=(1,0,1))
            self.branch2_4=Conv(in_channel=middle_channel_list[2],out_channel=middle_channel_list[3],kernel_size=(3,3,1),stride=1,padding=(1,1,0))
            self.branch2_5=Conv(in_channel=middle_channel_list[3],out_channel=out_channel_list[1],kernel_size=3,stride=1,padding=1)
            self.branch3_1=nn.MaxPool3d(kernel_size=3,stride=1,padding=1)
        else:
            self.branch1_1 = Conv(in_channel=in_channel, out_channel=middle_channel_list[0], kernel_size=1, stride=1,
                                   padding=0)
            self.branch1_2 = Conv(in_channel=middle_channel_list[0], out_channel=out_channel_list[0], kernel_size=3,
                                   stride=2, padding=1)
            self.branch2_1 = Conv(in_channel=in_channel, out_channel=middle_channel_list[1], kernel_size=1, stride=1,
                                   padding=0)
            self.branch2_2 = Conv(in_channel=middle_channel_list[1], out_channel=middle_channel_list[2],
                                   kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
            self.branch2_3 = Conv(in_channel=middle_channel_list[2], out_channel=middle_channel_list[3],
                                   kernel_size=(3, 1, 3), stride=1, padding=(1, 0, 1))
            self.branch2_4 = Conv(in_channel=middle_channel_list[2], out_channel=middle_channel_list[3],
                                     kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0))
            self.branch2_5 = Conv(in_channel=middle_channel_list[3], out_channel=out_channel_list[1], kernel_size=3,
                                   stride=2, padding=1)
            self.branch3_1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self,x):
        output1=self.branch1_2(self.branch1_1(x))
        output2=self.branch2_5(self.branch2_4(self.branch2_3(self.branch2_2(self.branch2_1(x)))))
        output3=self.branch3_1(x)
        # print(output1.shape, output2.shape, output3.shape)
        return torch.cat((output1,output2,output3),dim=1)

class DFMAS(nn.Module):
    def __init__(self, in_ch):
        super(DFMAS, self).__init__()
        self.in_ch3 = nn.Conv3d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.in_ch5 = nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.in_ch7 = nn.Conv3d(in_ch, in_ch, kernel_size=5, stride=1, padding=2, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm3d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.out_ch = nn.Conv3d(in_ch, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        o1 = self.in_ch3(x)
        o2 = self.in_ch5(x)
        o3 = self.in_ch7(x)
        out = o1 + o2 + o3
        out = self.bn(out)
        out = self.relu(out)
        out = self.out_ch(out)
        out = self.sigmoid(out)
        out_ch = x*out
        return out_ch


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
                # ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            else:
                input_channel = n_filters_out
                # ops.append(DFMAS(in_ch=input_channel))

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

            if i != n_stages-1:
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
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
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

class SideConv(nn.Module):
    def __init__(self, n_classes=1):
        super(SideConv, self).__init__()

        self.side5 = nn.Conv3d(256, n_classes, 1, padding=0)
        self.side4 = nn.Conv3d(128, n_classes, 1, padding=0)
        self.side3 = nn.Conv3d(64, n_classes, 1, padding=0)
        self.side2 = nn.Conv3d(32, n_classes, 1, padding=0)
        self.side1 = nn.Conv3d(16, n_classes, 1, padding=0)
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, stage_feat):
        x5, x5_up, x6_up, x7_up, x8_up = stage_feat[0], stage_feat[1], stage_feat[2], stage_feat[3], stage_feat[4]
        out5 = self.side5(x5)
        out5 = self.upsamplex2(out5)
        out5 = self.upsamplex2(out5)
        out5 = self.upsamplex2(out5)
        out5 = self.upsamplex2(out5)

        out4 = self.side4(x5_up)
        out4 = self.upsamplex2(out4)
        out4 = self.upsamplex2(out4)
        out4 = self.upsamplex2(out4)

        out3 = self.side3(x6_up)
        out3 = self.upsamplex2(out3)
        out3 = self.upsamplex2(out3)

        out2 = self.side2(x7_up)
        out2 = self.upsamplex2(out2)

        out1 = self.side1(x8_up)
        return [out5, out4, out3, out2, out1]

class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout
        self.sideconv = SideConv()

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)


        self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features, an=[]):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up_ori = x5_up + x4

        x6 = self.block_six(x5_up_ori)
        x6_up_ori = self.block_six_up(x6)
        x6_up = x6_up_ori + x3

        x7 = self.block_seven(x6_up)
        x7_up_ori = self.block_seven_up(x7)
        x7_up = x7_up_ori + x2

        x8 = self.block_eight(x7_up)
        x8_up_ori = self.block_eight_up(x8)
        x8_up = x8_up_ori + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out, [x5, x5_up, x6_up_ori, x7_up_ori, x8_up_ori]


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        
        features = self.encoder(input)
        # print(features[0].shape, features[1].shape, features[2].shape, features[3].shape, features[4].shape)
        out, stage_outr = self.decoder(features)
        deep_out = self.sideconv(stage_outr)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out, deep_out


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m,nn.ConvTranspose3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format
    model = VNet(n_channels=1, n_classes=1, normalization='batchnorm', has_dropout=False)
    input = torch.randn(1, 1, 112, 112, 80)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
