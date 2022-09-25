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
