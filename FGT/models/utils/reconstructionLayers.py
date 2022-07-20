# Two types of reconstructionl layers: 1. original residual layers, 2. residual layers with contrast and adaptive attention(CCA layer)
import torch
import torch.nn as nn
import torch.nn.init as init


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """

        Args:
            x: with shape of [b, c, t, h, w]

        Returns: processed features with shape [b, c, t, h, w]

        """
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        out = identity + out
        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html
        return out


class ResBlock_noBN_new(nn.Module):
    def __init__(self, nf):
        super(ResBlock_noBN_new, self).__init__()
        self.c1 = nn.Conv3d(nf, nf // 4, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.d1 = nn.Conv3d(nf // 4, nf // 4, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                            bias=True)  # dilation rate=1
        self.d2 = nn.Conv3d(nf // 4, nf // 4, kernel_size=(1, 3, 3), stride=1, padding=(0, 2, 2), dilation=(1, 2, 2),
                            bias=True)  # dilation rate=2
        self.d3 = nn.Conv3d(nf // 4, nf // 4, kernel_size=(1, 3, 3), stride=1, padding=(0, 4, 4), dilation=(1, 4, 4),
                            bias=True)  # dilation rate=4
        self.d4 = nn.Conv3d(nf // 4, nf // 4, kernel_size=(1, 3, 3), stride=1, padding=(0, 8, 8), dilation=(1, 8, 8),
                            bias=True)  # dilation rate=8
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.c2 = nn.Conv3d(nf, nf, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)

    def forward(self, x):
        output1 = self.act(self.c1(x))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)

        add1 = d1 + d2
        add2 = add1 + d3
        add3 = add2 + d4
        combine = torch.cat([d1, add1, add2, add3], dim=1)
        output2 = self.c2(self.act(combine))
        output = x + output2
        # remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html
        return output


class CCALayer(nn.Module):  #############################################3 new
    '''Residual block w/o BN
    --conv--contrast-conv--x---
      |    \--mean--|     |
      |___________________|

    '''

    def __init__(self, nf=64):
        super(CCALayer, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_du = nn.Sequential(
            nn.Conv2d(nf, 4, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, nf, 1, padding=0, bias=True),
            nn.Tanh()  # change from `Sigmoid` to `Tanh` to make the output between -1 and 1
        )
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # initialization
        initialize_weights([self.conv1, self.conv_du], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        out = self.contrast(out) + self.avg_pool(out)
        out_channel = self.conv_du(out)
        out_channel = out_channel * out
        out_last = out_channel + identity

        return out_last


def mean_channels(F):
    assert (F.dim() == 4), 'Your dim is {} bit not 4'.format(F.dim())
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))  # 对每一个channel都求其特征图的高和宽的平均值


def stdv_channels(F):
    assert F.dim() == 4, 'Your dim is {} bit not 4'.format(F.dim())
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
