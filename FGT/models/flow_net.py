import torch
import torch.nn.functional as F
import torch.nn as nn
import functools
from .BaseNetwork import BaseNetwork
from models.utils.reconstructionLayers import make_layer, ResidualBlock_noBN


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.net = P3DNet(config['num_flows'], config['cnum'], config['in_channel'], config['PASSMASK'],
                          config['use_residual'],
                          config['resBlocks'], config['use_bias'], config['conv_type'], config['multi_out'],
                          config['init_weights'])

    def forward(self, flows, masks, edges=None):
        ret = self.net(flows, masks, edges)
        return ret


class P3DNet(BaseNetwork):
    def __init__(self, num_flows, num_feats, in_channels, passmask, use_residual, res_blocks,
                 use_bias, conv_type, multi_out, init_weights):
        super().__init__(conv_type)
        self.passmask = passmask
        self.multi_out = multi_out
        self.encoder2 = nn.Sequential(
            nn.ReplicationPad2d(2),
            self.ConvBlock2d(in_channels, num_feats, kernel_size=5, stride=1, padding=0, bias=use_bias, norm=None),
            self.ConvBlock2d(num_feats, num_feats * 2, kernel_size=3, stride=2, padding=1, bias=use_bias, norm=None)
        )
        self.encoder4 = nn.Sequential(
            self.ConvBlock2d(num_feats * 2, num_feats * 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                             norm=None),
            self.ConvBlock2d(num_feats * 2, num_feats * 4, kernel_size=3, stride=2, padding=1, bias=use_bias, norm=None)
        )
        residualBlock = functools.partial(ResidualBlock_noBN, nf=num_feats * 4)
        self.res_blocks = make_layer(residualBlock, res_blocks)
        self.resNums = res_blocks
        # dilation convolution to enlarge the receptive field
        self.middle = nn.Sequential(
            self.ConvBlock2d(num_feats * 4, num_feats * 4, kernel_size=3, stride=1, padding=8, bias=use_bias,
                             dilation=8, norm=None),
            self.ConvBlock2d(num_feats * 4, num_feats * 4, kernel_size=3, stride=1, padding=4, bias=use_bias,
                             dilation=4, norm=None),
            self.ConvBlock2d(num_feats * 4, num_feats * 4, kernel_size=3, stride=1, padding=2, bias=use_bias,
                             dilation=2, norm=None),
            self.ConvBlock2d(num_feats * 4, num_feats * 4, kernel_size=3, stride=1, padding=1, bias=use_bias,
                             dilation=1, norm=None),
        )
        self.decoder2 = nn.Sequential(
            self.DeconvBlock2d(num_feats * 8, num_feats * 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                               norm=None),
            self.ConvBlock2d(num_feats * 2, num_feats * 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                             norm=None),
            self.ConvBlock2d(num_feats * 2, num_feats * 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                             norm=None)
        )
        if self.multi_out:
            self.out4 = self.ConvBlock2d(num_feats * 4, 2, kernel_size=1, stride=1, padding=0, bias=use_bias, norm=None)
            self.out2 = self.ConvBlock2d(num_feats * 2, 2, kernel_size=1, stride=1, padding=0, bias=use_bias, norm=None)
        self.decoder = nn.Sequential(
            self.DeconvBlock2d(num_feats * 4, num_feats, kernel_size=3, stride=1, padding=1, bias=use_bias,
                               norm=None),
            self.ConvBlock2d(num_feats, num_feats // 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                             norm=None),
            self.ConvBlock2d(num_feats // 2, 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                             norm=None)
        )
        self.edgeDetector = EdgeDetection(conv_type)
        if init_weights:
            self.init_weights()

    def forward(self, flows, masks, edges=None):
        """
        Args:
            flows: The diffused optical flows (initial inpainted optical flows for flow warping), woth shape [b, 2, t, h, w]
            masks: tensor, with shape [b, 1, t, h, w], the middle of t is the target flow tensor
            edges: tensor, with shape [b, 1, t, h, w]

        Returns: middle of the target flow

        """
        if self.passmask:
            inputs = torch.cat((flows, masks), dim=1)
        else:
            inputs = flows
        if edges is not None:
            inputs = torch.cat((inputs, edges), dim=1)
        e2 = self.encoder2(inputs)
        e4 = self.encoder4(e2)
        if self.resNums > 0:
            e4_res = self.res_blocks(e4)
        else:
            e4_res = e4
        c_e4_filled = self.middle(e4_res)
        if self.multi_out:
            out4 = self.out4(c_e4_filled)
        c_e4 = torch.cat((c_e4_filled, e4), dim=1)
        c_e2Post = self.decoder2(c_e4)
        if self.multi_out:
            out2 = self.out2(c_e2Post)
        c_e2 = torch.cat((c_e2Post, e2), dim=1)
        output = self.decoder(c_e2)
        edge = self.edgeDetector(output)
        if self.multi_out:
            return output, out4, out2, edge
        return output, edge


class EdgeDetection(BaseNetwork):
    def __init__(self, conv_type, in_channels=2, out_channels=1, mid_channels=16):
        super(EdgeDetection, self).__init__(conv_type)
        self.projection = self.ConvBlock2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                           padding=1, norm=None)
        self.mid_layer_1 = self.ConvBlock2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3,
                                            stride=1, padding=1, norm=None)
        self.mid_layer_2 = self.ConvBlock2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3,
                                            stride=1, padding=1, activation=None, norm=None)
        self.l_relu = nn.LeakyReLU()
        self.out_layer = self.ConvBlock2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1,
                                          activation=None, norm=None)

    def forward(self, flow):
        flow = self.projection(flow)
        edge = self.mid_layer_1(flow)
        edge = self.mid_layer_2(edge)
        edge = self.l_relu(flow + edge)
        edge = self.out_layer(edge)
        edge = torch.sigmoid(edge)
        return edge


