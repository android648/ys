import torch
import torch.nn as nn
import torch.nn.functional as F
#(W-F+2P)/S+1

def Conv1(in_channel,out_channel,kernel_size,**kwargs):
    return nn.Sequential(nn.Conv3d(in_channel,out_channel,kernel_size,**kwargs),
                         nn.BatchNorm3d(out_channel),
                         nn.LeakyReLU())

class InsertB(nn.Module):
    def __init__(self,n,in_channel,out_channel_list,middle_channel_list):
        super(InsertB, self).__init__()
        if n == 1:
            self.branch1_1=Conv1(in_channel=in_channel,out_channel=middle_channel_list[0],kernel_size=1,stride=1,padding=0)
            self.branch1_2=Conv1(in_channel=middle_channel_list[0],out_channel=out_channel_list[0],kernel_size=3,stride=1,padding=1)
            self.branch2_1=Conv1(in_channel=in_channel,out_channel=middle_channel_list[1],kernel_size=1,stride=1,padding=0)
            self.branch2_2=Conv1(in_channel=middle_channel_list[1],out_channel=middle_channel_list[2],kernel_size=(1,3,3),stride=1,padding=(0,1,1))
            self.branch2_3=Conv1(in_channel=middle_channel_list[2],out_channel=middle_channel_list[3],kernel_size=(3,1,3),stride=1,padding=(1,0,1))
            self.branch2_3_1=Conv1(in_channel=middle_channel_list[2],out_channel=middle_channel_list[3],kernel_size=(3,3,1),stride=1,padding=(1,1,0))
            self.branch2_4=Conv1(in_channel=middle_channel_list[3],out_channel=out_channel_list[1],kernel_size=3,stride=1,padding=1)
            self.branch3_1=nn.MaxPool3d(kernel_size=3,stride=1,padding=1)
        else:
            self.branch1_1 = Conv1(in_channel=in_channel, out_channel=middle_channel_list[0], kernel_size=1, stride=1,
                                   padding=0)
            self.branch1_2 = Conv1(in_channel=middle_channel_list[0], out_channel=out_channel_list[0], kernel_size=3,
                                   stride=2, padding=1)
            self.branch2_1 = Conv1(in_channel=in_channel, out_channel=middle_channel_list[1], kernel_size=1, stride=1,
                                   padding=0)
            self.branch2_2 = Conv1(in_channel=middle_channel_list[1], out_channel=middle_channel_list[2],
                                   kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
            self.branch2_3 = Conv1(in_channel=middle_channel_list[2], out_channel=middle_channel_list[3],
                                   kernel_size=(3, 1, 3), stride=1, padding=(1, 0, 1))
            self.branch2_3_1 = Conv1(in_channel=middle_channel_list[2], out_channel=middle_channel_list[3],
                                     kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0))
            self.branch2_4 = Conv1(in_channel=middle_channel_list[3], out_channel=out_channel_list[1], kernel_size=3,
                                   stride=2, padding=1)
            self.branch3_1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self,x):
        output1=self.branch1_2(self.branch1_1(x))
        output2=self.branch2_4(self.branch2_3_1(self.branch2_3(self.branch2_2(self.branch2_1(x)))))
        output3=self.branch3_1(x)
        print(output1.shape, output2.shape, output3.shape)
        return torch.cat((output1,output2,output3),dim=1)

        # return output1+output2

class Vnet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Vnet, self).__init__()
        self.block_one = InsertB(1, n_channels, [7,8], middle_channel_list=[7,8,8,8])
        self.block_two = InsertB(2, n_filters, [8,8], middle_channel_list=[8,8,8,8])
        self.block_three = InsertB(3, n_filters * 2, [16,16], middle_channel_list=[16,16,16,16])
        self.block_four = InsertB(4, n_filters * 4, [32,32], middle_channel_list=[32,32,32,32])
        self.block_five = InsertB(5, n_filters * 8, [64,64], middle_channel_list=[64,64,64,64])

    def forward(self, input):
        x1 = self.block_one(input)
        print(x1.shape)
        x2 = self.block_two(x1)
        print(x2.shape)
        x3 = self.block_three(x2)
        print(x3.shape)
        x4 = self.block_four(x3)
        print(x4.shape)
        x5 = self.block_five(x4)
        print(x5.shape)
        return x1, x2, x3, x4, x5


if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format
    model = Vnet(n_channels=1, n_classes=1, normalization='batchnorm', has_dropout=False)
    input = torch.randn(1, 1, 112, 112, 80)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
