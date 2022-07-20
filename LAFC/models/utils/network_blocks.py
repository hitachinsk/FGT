import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class VanillaConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True)
    ):

        super().__init__()

        self.padding = tuple(((np.array(kernel_size) - 1) * np.array(dilation)) // 2) if padding == -1 else padding
        self.featureConv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, self.padding, dilation, groups, bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm3d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm3d(out_channels, track_running_stats=True)
        elif norm == "SN":
            self.norm = None
            self.featureConv = nn.utils.spectral_norm(self.featureConv)
        else:
            self.norm = None

        self.activation = activation

    def forward(self, xs):
        out = self.featureConv(xs)
        if self.activation:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class VanillaDeconv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
        scale_factor=2
    ):
        super().__init__()
        self.conv = VanillaConv(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation)
        self.scale_factor = scale_factor

    def forward(self, xs):
        xs_resized = F.interpolate(xs, scale_factor=(1, self.scale_factor, self.scale_factor))
        return self.conv(xs_resized)


class GatedConv(VanillaConv):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True)
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation
        )
        self.gatingConv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, self.padding, dilation, groups, bias)
        if norm == 'SN':
            self.gatingConv = nn.utils.spectral_norm(self.gatingConv)
        self.sigmoid = nn.Sigmoid()
        self.store_gated_values = False

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        out = self.sigmoid(mask)
        if self.store_gated_values:
            self.gated_values = out.detach().cpu()
        return out

    def forward(self, xs):
        gating = self.gatingConv(xs)
        feature = self.featureConv(xs)
        if self.activation:
            feature = self.activation(feature)
        out = self.gated(gating) * feature
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class GatedDeconv(VanillaDeconv):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
        scale_factor=2
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, scale_factor
        )
        self.conv = GatedConv(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation)


class PartialConv(VanillaConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation
        )
        self.mask_sum_conv = self.module.Conv3d(1, 1, kernel_size,
                                                stride, padding, dilation, groups, False)
        nn.init.constant_(self.mask_sum_conv.weight, 1.0)

        # mask conv needs not update
        for param in self.mask_sum_conv.parameters():
            param.requires_grad = False

        if norm == "SN":
            self.featureConv = nn.utils.spectral_norm(self.featureConv)
            raise NotImplementedError(f"Norm type {norm} not implemented")

    def forward(self, input_tuple):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # output = W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0), if sum(M) != 0
        #        = 0, if sum(M) == 0
        inp, mask = input_tuple

        # C(M .* X)
        output = self.featureConv(mask * inp)

        # C(0) = b
        if self.featureConv.bias is not None:
            output_bias = self.featureConv.bias.view(1, -1, 1, 1, 1)
        else:
            output_bias = torch.zeros([1, 1, 1, 1, 1]).to(inp.device)

        # D(M) = sum(M)
        with torch.no_grad():
            mask_sum = self.mask_sum_conv(mask)

        # find those sum(M) == 0
        no_update_holes = (mask_sum == 0)

        # Just to prevent devided by 0
        mask_sum_no_zero = mask_sum.masked_fill_(no_update_holes, 1.0)

        # output = [C(M .* X) – C(0)] / D(M) + C(0), if sum(M) != 0
        #        = 0, if sum (M) == 0
        output = (output - output_bias) / mask_sum_no_zero + output_bias
        output = output.masked_fill_(no_update_holes, 0.0)

        # create a new mask with only 1 or 0
        new_mask = torch.ones_like(mask_sum)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        if self.activation is not None:
            output = self.activation(output)
        if self.norm is not None:
            output = self.norm_layer(output)
        return output, new_mask


class PartialDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
                 scale_factor=2):
        super().__init__()
        self.conv = PartialConv(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input_tuple):
        inp, mask = input_tuple
        inp_resized = F.interpolate(inp, scale_factor=(1, self.scale_factor, self.scale_factor))
        with torch.no_grad():
            mask_resized = F.interpolate(mask, scale_factor=(1, self.scale_factor, self.scale_factor))
        return self.conv((inp_resized, mask_resized))