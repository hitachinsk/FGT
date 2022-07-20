import torch
import torch.nn as nn
import torch.nn.functional as F

networks = ['BaseNetwork', 'Discriminator', 'ASPP']

# Base model borrows from PEN-NET
# https://github.com/researchmm/PEN-Net-for-Inpainting
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


# temporal patch gan: from Free-form Video Inpainting with 3D Gated Convolution and Temporal PatchGAN in 2019 ICCV
# todo: debug this model
class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 64

        self.conv = nn.Sequential(
            DisBuildingBlock(in_channel=in_channels, out_channel=nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                             padding=1, use_spectral_norm=use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            DisBuildingBlock(in_channel=nf * 1, out_channel=nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                             padding=(1, 2, 2), use_spectral_norm=use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            DisBuildingBlock(in_channel=nf * 2, out_channel=nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                             padding=(1, 2, 2), use_spectral_norm=use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            DisBuildingBlock(in_channel=nf * 4, out_channel=nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                             padding=(1, 2, 2), use_spectral_norm=use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            DisBuildingBlock(in_channel=nf * 4, out_channel=nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                             padding=(1, 2, 2), use_spectral_norm=use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5),
                      stride=(1, 2, 2), padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # B, C, T, H, W = xs.shape
        feat = self.conv(xs)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        return feat


class DisBuildingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, use_spectral_norm):
        super(DisBuildingBlock, self).__init__()
        self.block = self._getBlock(in_channel, out_channel, kernel_size, stride, padding, use_spectral_norm)

    def _getBlock(self, in_channel, out_channel, kernel_size, stride, padding, use_spectral_norm):
        feature_conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size,
                                 stride=stride, padding=padding, bias=not use_spectral_norm)
        if use_spectral_norm:
            feature_conv = nn.utils.spectral_norm(feature_conv)
        return feature_conv

    def forward(self, inputs):
        out = self.block(inputs)
        return out


class ASPP(nn.Module):
    def __init__(self, input_channels, output_channels, rate=[1, 2, 4, 8]):
        super(ASPP, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.rate = rate
        for i in range(len(rate)):
            self.__setattr__('conv{}'.format(str(i).zfill(2)), nn.Sequential(
                nn.Conv2d(input_channels, output_channels // len(rate), kernel_size=3, dilation=rate[i],
                          padding=rate[i]),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))

    def forward(self, inputs):
        inputs = inputs / len(self.rate)
        tmp = []
        for i in range(len(self.rate)):
            tmp.append(self.__getattr__('conv{}'.format(str(i).zfill(2)))(inputs))
        output = torch.cat(tmp, dim=1)
        return output


class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=False, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, inputs):
        x = self.conv2d(inputs)
        mask = self.mask_conv2d(inputs)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class GatedDeConv2dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=False, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv2dWithActivation, self).__init__()
        self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, inputs):
        #print(input.size())
        x = F.interpolate(inputs, scale_factor=self.scale_factor)
        return self.conv2d(x)


class SNGatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convolution with spetral normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=False, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNGatedConv2dWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation
        self.batch_norm = batch_norm
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.mask_conv2d = torch.nn.utils.spectral_norm(self.mask_conv2d)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, inputs):
        x = self.conv2d(inputs)
        mask = self.mask_conv2d(inputs)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class SNGatedDeConv2dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=False, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNGatedDeConv2dWithActivation, self).__init__()
        self.conv2d = SNGatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, inputs):
        x = F.interpolate(inputs, scale_factor=2)
        return self.conv2d(x)


class GatedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv3d, self).__init__()
        self.input_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.gating_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups, bias)
        self.activation = activation

    def forward(self, inputs):
        feature = self.input_conv(inputs)
        if self.activation:
            feature = self.activation(feature)
        gating = torch.sigmoid(self.gating_conv(inputs))
        return feature * gating


class GatedDeconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale_factor, dilation=1, groups=1, bias=True, activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.scale_factor = scale_factor
        self.deconv = GatedConv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, activation)

    def forward(self, inputs):
        inputs = F.interpolate(inputs, scale_factor=(1, self.scale_factor, self.scale_factor))
        return self.deconv(inputs)

