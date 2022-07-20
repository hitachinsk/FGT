from .utils.network_blocks import *
from .utils.network_blocks_2d import *


class BaseNetwork(nn.Module):
    def __init__(self, conv_type):
        super(BaseNetwork, self).__init__()
        self.conv_type = conv_type
        if conv_type == 'gated':
            self.ConvBlock = GatedConv
            self.DeconvBlock = GatedDeconv
            self.ConvBlock2d = GatedConv2d
            self.DeconvBlock2d = GatedDeconv2d
        if conv_type == 'partial':
            self.ConvBlock = PartialConv
            self.DeconvBlock = PartialDeconv
            self.ConvBlock2d = PartialConv2d
            self.DeconvBlock2d = PartialDeconv2d
        if conv_type == 'vanilla':
            self.ConvBlock = VanillaConv
            self.DeconvBlock = VanillaDeconv
            self.ConvBlock2d = VanillaConv2d
            self.DeconvBlock2d = VanillaDeconv2d

    def init_weights(self, init_type='kaiming', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
