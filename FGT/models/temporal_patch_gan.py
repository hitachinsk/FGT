# temporal patch GAN to maintain the temporal consecutive of the flows
import torch
import torch.nn as nn
from .BaseNetwork import BaseNetwork


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, conv_type, dist_cnum, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        """

        Args:
            in_channels: The input channels of the discriminator
            use_sigmoid: Whether to use sigmoid for the base network (true for the nsgan)
            use_spectral_norm: The usage of the spectral norm: always be true for the stability of GAN
            init_weights: always be True
        """
        super(Discriminator, self).__init__(conv_type)
        self.use_sigmoid = use_sigmoid
        nf = dist_cnum

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels, out_channels=nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(in_channels=nf * 1, out_channels=nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(in_channels=nf * 4, out_channels=nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(in_channels=nf * 4, out_channels=nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels=nf * 4, out_channels=nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                      padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs, t):
        """

        Args:
            xs: Input feature, with shape of [bt, c, h, w]

        Returns: The discriminative map from the GAN

        """
        bt, c, h, w = xs.shape
        b = bt // t
        xs = xs.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        feat = self.conv(xs)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # [b, t, c, h, w]
        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module
