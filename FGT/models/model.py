from models.BaseNetwork import BaseNetwork
from models.transformer_base.ffn_base import FusionFeedForward
from models.transformer_base.attention_flow import SWMHSA_depthGlobalWindowConcatLN_qkFlow_reweightFlow
from models.transformer_base.attention_base import TMHSA

import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.net = FGT(config['tw'], config['sw'], config['gd'], config['input_resolution'], config['in_channel'],
                        config['cnum'], config['flow_inChannel'], config['flow_cnum'], config['frame_hidden'],
                        config['flow_hidden'], config['PASSMASK'],
                        config['numBlocks'], config['kernel_size'], config['stride'], config['padding'],
                        config['num_head'], config['conv_type'], config['norm'],
                        config['use_bias'], config['ape'],
                        config['mlp_ratio'], config['drop'], config['init_weights'])

    def forward(self, frames, flows, masks):
        ret = self.net(frames, flows, masks)
        return ret


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, h, w = x.size()
        h, w = h // 4, w // 4
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)
        return out


class AddPosEmb(nn.Module):
    def __init__(self, h, w, in_channels, out_channels):
        super(AddPosEmb, self).__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True, groups=out_channels)
        self.h, self.w = h, w

    def forward(self, x, h=0, w=0):
        B, N, C = x.shape
        if h == 0 and w == 0:
            assert N == self.h * self.w, 'Wrong input size'
        else:
            assert N == h * w, 'Wrong input size during inference'
        feat_token = x
        if h == 0 and w == 0:
            cnn_feat = feat_token.transpose(1, 2).view(B, C, self.h, self.w)
        else:
            cnn_feat = feat_token.transpose(1, 2).view(B, C, h, w)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        return x


class Vec2Patch(nn.Module):
    def __init__(self, channel, hidden, output_size, kernel_size, stride, padding):
        super(Vec2Patch, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.restore = nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x, output_h=0, output_w=0):
        feat = self.embedding(x)
        feat = feat.permute(0, 2, 1)
        if output_h != 0 or output_w != 0:
            feat = F.fold(feat, output_size=(output_h, output_w), kernel_size=self.kernel_size, stride=self.stride,
                          padding=self.padding)
        else:
            feat = self.restore(feat)
        return feat


class TemporalTransformer(nn.Module):
    def __init__(self, token_size, frame_hidden, num_heads, t_groupSize, mlp_ratio, dropout, n_vecs,
                 t2t_params):
        super(TemporalTransformer, self).__init__()
        self.attention = TMHSA(token_size=token_size, group_size=t_groupSize, d_model=frame_hidden, head=num_heads,
                               p=dropout)
        self.ffn = FusionFeedForward(frame_hidden, mlp_ratio, n_vecs, t2t_params, p=dropout)
        self.norm1 = nn.LayerNorm(frame_hidden)
        self.norm2 = nn.LayerNorm(frame_hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, t, h, w, output_size):
        token_size = h * w
        s = self.norm1(x)
        x = x + self.dropout(self.attention(s, t, h, w))
        y = self.norm2(x)
        x = x + self.ffn(y, token_size, output_size[0], output_size[1])
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, token_size, frame_hidden, flow_hidden, num_heads, s_windowSize, g_downSize, mlp_ratio,
                 dropout, n_vecs, t2t_params):
        super(SpatialTransformer, self).__init__()
        self.attention = SWMHSA_depthGlobalWindowConcatLN_qkFlow_reweightFlow(token_size=token_size, window_size=s_windowSize,
                                                                kernel_size=g_downSize, d_model=frame_hidden,
                                                                flow_dModel=flow_hidden, head=num_heads, p=dropout)
        self.ffn = FusionFeedForward(frame_hidden, mlp_ratio, n_vecs, t2t_params, p=dropout)
        self.norm = nn.LayerNorm(frame_hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, f, t, h, w, output_size):
        token_size = h * w
        x = x + self.dropout(self.attention(x, f, t, h, w))
        y = self.norm(x)
        x = x + self.ffn(y, token_size, output_size[0], output_size[1])
        return x


class TransformerBlock(nn.Module):
    def __init__(self, token_size, frame_hidden, flow_hidden, num_heads, t_groupSize, s_windowSize, g_downSize,
                 mlp_ratio,
                 dropout, n_vecs,
                 t2t_params):
        super(TransformerBlock, self).__init__()
        self.t_transformer = TemporalTransformer(token_size=token_size, frame_hidden=frame_hidden, num_heads=num_heads,
                                                 t_groupSize=t_groupSize, mlp_ratio=mlp_ratio,
                                                 dropout=dropout, n_vecs=n_vecs,
                                                 t2t_params=t2t_params)  # temporal multi-head self attention
        self.s_transformer = SpatialTransformer(token_size=token_size, frame_hidden=frame_hidden,
                                                flow_hidden=flow_hidden, num_heads=num_heads, s_windowSize=s_windowSize,
                                                g_downSize=g_downSize, mlp_ratio=mlp_ratio,
                                                dropout=dropout, n_vecs=n_vecs, t2t_params=t2t_params)

    def forward(self, inputs):
        x, f, t = inputs['x'], inputs['f'], inputs['t']
        h, w = inputs['h'], inputs['w']
        output_size = inputs['output_size']
        x = self.t_transformer(x, t, h, w, output_size)
        x = self.s_transformer(x, f, t, h, w, output_size)
        return {'x': x, 'f': f, 't': t, 'h': h, 'w': w, 'output_size': output_size}


class Decoder(BaseNetwork):
    def __init__(self, conv_type, in_channels, out_channels, use_bias, norm=None):
        super(Decoder, self).__init__(conv_type)
        self.layer1 = self.DeconvBlock(in_channels, in_channels, kernel_size=3, padding=1, norm=norm,
                                       bias=use_bias)
        self.layer2 = self.ConvBlock(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, norm=norm,
                                     bias=use_bias)
        self.layer3 = self.DeconvBlock(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, norm=norm,
                                       bias=use_bias)
        self.final = self.ConvBlock(in_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, norm=norm,
                                    bias=use_bias, activation=None)

    def forward(self, features):
        feat1 = self.layer1(features)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        output = self.final(feat3)
        return output


class FGT(BaseNetwork):
    def __init__(self, t_groupSize, s_windowSize, g_downSize, input_resolution, in_channels, cnum, flow_inChannel,
                 flow_cnum,
                 frame_hidden, flow_hidden, passmask, numBlocks, kernel_size, stride, padding, num_heads, conv_type,
                 norm, use_bias, ape, mlp_ratio=4, drop=0, init_weights=True):
        super(FGT, self).__init__(conv_type)
        self.in_channels = in_channels
        self.passmask = passmask
        self.ape = ape
        self.frame_endoder = Encoder(in_channels)
        self.flow_encoder = nn.Sequential(
            nn.ReplicationPad2d(2),
            self.ConvBlock(flow_inChannel, flow_cnum, kernel_size=5, stride=1, padding=0, bias=use_bias, norm=norm),
            self.ConvBlock(flow_cnum, flow_cnum * 2, kernel_size=3, stride=2, padding=1, bias=use_bias, norm=norm),
            self.ConvBlock(flow_cnum * 2, flow_cnum * 2, kernel_size=3, stride=1, padding=1, bias=use_bias, norm=norm),
            self.ConvBlock(flow_cnum * 2, flow_cnum * 2, kernel_size=3, stride=2, padding=1, bias=use_bias, norm=norm)
        )
        # patch to vector operation
        self.patch2vec = nn.Conv2d(cnum * 2, frame_hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.f_patch2vec = nn.Conv2d(flow_cnum * 2, flow_hidden, kernel_size=kernel_size, stride=stride,
                                     padding=padding)
        # initialize transformer blocks for frame completion
        n_vecs = 1
        token_size = []
        output_shape = (input_resolution[0] // 4, input_resolution[1] // 4)
        for i, d in enumerate(kernel_size):
            token_nums = int((output_shape[i] + 2 * padding[i] - kernel_size[i]) / stride[i] + 1)
            n_vecs *= token_nums
            token_size.append(token_nums)
        # Add positional embedding to the encode features
        if self.ape:
            self.add_pos_emb = AddPosEmb(token_size[0], token_size[1], frame_hidden, frame_hidden)
        self.token_size = token_size
        # initialize transformer blocks
        blocks = []
        t2t_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'output_size': output_shape}
        for i in range(numBlocks // 2 - 1):
            layer = TransformerBlock(token_size, frame_hidden, flow_hidden, num_heads, t_groupSize, s_windowSize,
                                     g_downSize, mlp_ratio, drop, n_vecs, t2t_params)
            blocks.append(layer)
        self.first_t_transformer = TemporalTransformer(token_size, frame_hidden, num_heads, t_groupSize, mlp_ratio,
                                                       drop, n_vecs, t2t_params)
        self.first_s_transformer = SpatialTransformer(token_size, frame_hidden, flow_hidden, num_heads, s_windowSize,
                                                      g_downSize, mlp_ratio, drop, n_vecs, t2t_params)
        self.transformer = nn.Sequential(*blocks)
        # vector to patch operation
        self.vec2patch = Vec2Patch(cnum * 2, frame_hidden, output_shape, kernel_size, stride, padding)
        # decoder
        self.decoder = Decoder(conv_type, cnum * 2, 3, use_bias, norm)

        if init_weights:
            self.init_weights()

    def forward(self, masked_frames, flows, masks):
        b, t, c, h, w = masked_frames.shape
        cf = flows.shape[2]
        output_shape = (h // 4, w // 4)
        if self.passmask:
            inputs = torch.cat((masked_frames, masks), dim=2)
        else:
            inputs = masked_frames
        inputs = inputs.view(b * t, self.in_channels, h, w)
        flows = flows.view(b * t, cf, h, w)
        enc_feats = self.frame_endoder(inputs)
        flow_feats = self.flow_encoder(flows)
        trans_feat = self.patch2vec(enc_feats)
        flow_patches = self.f_patch2vec(flow_feats)
        _, c, h, w = trans_feat.shape
        cf = flow_patches.shape[1]
        if h != self.token_size[0] or w != self.token_size[1]:
            new_h, new_w = h, w
        else:
            new_h, new_w = 0, 0
            output_shape = (0, 0)
        trans_feat = trans_feat.view(b * t, c, -1).permute(0, 2, 1)
        flow_patches = flow_patches.view(b * t, cf, -1).permute(0, 2, 1)
        trans_feat = self.first_t_transformer(trans_feat, t, new_h, new_w, output_shape)
        trans_feat = self.add_pos_emb(trans_feat, new_h, new_w)
        trans_feat = self.first_s_transformer(trans_feat, flow_patches, t, new_h, new_w, output_shape)
        inputs_trans_feat = {'x': trans_feat, 'f': flow_patches, 't': t, 'h': new_h, 'w': new_w,
                             'output_size': output_shape}
        trans_feat = self.transformer(inputs_trans_feat)['x']
        trans_feat = self.vec2patch(trans_feat, output_shape[0], output_shape[1])
        enc_feats = enc_feats + trans_feat

        output = self.decoder(enc_feats)
        output = torch.tanh(output)
        return output

