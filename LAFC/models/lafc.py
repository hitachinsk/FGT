import torch
import torch.nn as nn
from .BaseNetwork import BaseNetwork


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.net = P3DNet(config['num_flows'], config['cnum'], config['in_channel'], config['PASSMASK'],
                          config['use_residual'],
                          config['resBlocks'], config['use_bias'], config['conv_type'], config['init_weights'])

    def forward(self, flows, masks, edges=None):
        ret = self.net(flows, masks, edges)
        return ret


class P3DNet(BaseNetwork):
    def __init__(self, num_flows, num_feats, in_channels, passmask, use_residual, res_blocks,
                 use_bias, conv_type, init_weights):
        super().__init__(conv_type)
        self.passmask = passmask
        self.encoder2 = nn.Sequential(
            nn.ReplicationPad3d((2, 2, 2, 2, 0, 0)),
            P3DBlock(in_channels, num_feats, kernel_size=5, stride=1, padding=0, bias=use_bias, conv_type=conv_type,
                     norm=None, use_residual=0),
            P3DBlock(num_feats, num_feats * 2, kernel_size=3, stride=2, padding=1, bias=use_bias, conv_type=conv_type,
                     norm=None, use_residual=0)
        )
        self.encoder4 = nn.Sequential(
            P3DBlock(num_feats * 2, num_feats * 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                     conv_type=conv_type, norm=None, use_residual=use_residual),
            P3DBlock(num_feats * 2, num_feats * 4, kernel_size=3, stride=2, padding=1, bias=use_bias,
                     conv_type=conv_type, norm=None, use_residual=0)
        )
        residual_blocks = []
        self.resNums = res_blocks
        base_residual_block = P3DBlock(num_feats * 4, num_feats * 4, kernel_size=3, stride=1, padding=1, bias=use_bias,
                                       conv_type=conv_type,
                                       norm=None, use_residual=1)
        for _ in range(res_blocks):
            residual_blocks.append(base_residual_block)
        self.res_blocks = nn.Sequential(*residual_blocks)
        self.condense2 = self.ConvBlock(num_feats * 2, num_feats * 2, kernel_size=(num_flows, 1, 1), stride=1,
                                        padding=0,
                                        bias=use_bias, norm=None)
        self.condense4_pre = self.ConvBlock(num_feats * 4, num_feats * 4, kernel_size=(num_flows, 1, 1), stride=1,
                                            padding=0,
                                            bias=use_bias, norm=None)
        self.condense4_post = self.ConvBlock(num_feats * 4, num_feats * 4, kernel_size=(num_flows, 1, 1), stride=1,
                                             padding=0,
                                             bias=use_bias, norm=None)
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
        self.decoder = nn.Sequential(
            self.DeconvBlock2d(num_feats * 4, num_feats, kernel_size=3, stride=1, padding=1, bias=use_bias,
                               norm=None),
            self.ConvBlock2d(num_feats, num_feats // 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                             norm=None),
            self.ConvBlock2d(num_feats // 2, 2, kernel_size=3, stride=1, padding=1, bias=use_bias,
                             norm=None, activation=None)
        )
        self.edgeDetector = EdgeDetection(conv_type)
        if init_weights:
            self.init_weights()

    def forward(self, flows, masks, edges=None):
        if self.passmask:
            inputs = torch.cat((flows, masks), dim=1)
        else:
            inputs = flows
        if edges is not None:
            inputs = torch.cat((inputs, edges), dim=1)
        e2 = self.encoder2(inputs)
        c_e2Pre = self.condense2(e2).squeeze(2)
        e4 = self.encoder4(e2)
        c_e4Pre = self.condense4_pre(e4).squeeze(2)
        if self.resNums > 0:
            e4 = self.res_blocks(e4)
        c_e4Post = self.condense4_post(e4).squeeze(2)
        assert len(c_e4Post.shape) == 4, 'Wrong with the c_e4 shape: {}'.format(len(c_e4Post.shape))
        c_e4_filled = self.middle(c_e4Post)
        c_e4 = torch.cat((c_e4_filled, c_e4Pre), dim=1)
        c_e2Post = self.decoder2(c_e4)
        c_e2 = torch.cat((c_e2Post, c_e2Pre), dim=1)
        output = self.decoder(c_e2)
        edge = self.edgeDetector(output)
        return output, edge


class P3DBlock(BaseNetwork):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, conv_type, norm, use_residual):
        super().__init__(conv_type)
        self.conv1 = self.ConvBlock(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size),
                                    stride=(1, stride, stride), padding=(0, padding, padding),
                                    bias=bias, norm=norm)
        self.conv2 = self.ConvBlock(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1,
                                    padding=(1, 0, 0), bias=bias, norm=norm)
        self.use_residual = use_residual

    def forward(self, feats):
        feat1 = self.conv1(feats)
        feat2 = self.conv2(feat1)
        if self.use_residual:
            output = feats + feat2
        else:
            output = feat2
        return output


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
