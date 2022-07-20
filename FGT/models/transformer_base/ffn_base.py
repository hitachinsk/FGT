import torch
from functools import reduce
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class FeedForward(nn.Module):
    def __init__(self, frame_hidden, mlp_ratio, n_vecs, t2t_params, p):
        """

        Args:
            frame_hidden: hidden size of frame features
            mlp_ratio: mlp ratio in the middle layer of the transformers
            n_vecs: number of vectors in the transformer
            t2t_params: dictionary -> {'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'output_size': output_shape}
            p: dropout rate, 0 by default
        """
        super(FeedForward, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(frame_hidden, frame_hidden * mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(frame_hidden * mlp_ratio, frame_hidden),
            nn.Dropout(p)
        )

    def forward(self, x, n_vecs=0, output_h=0, output_w=0):
        x = self.conv(x)
        return x


class FusionFeedForward(nn.Module):
    def __init__(self, frame_hidden, mlp_ratio, n_vecs, t2t_params, p):
        super(FusionFeedForward, self).__init__()
        self.kernel_shape = reduce((lambda x, y: x * y), t2t_params['kernel_size'])
        self.t2t_params = t2t_params
        hidden_size = self.kernel_shape * mlp_ratio
        self.conv1 = nn.Linear(frame_hidden, hidden_size)
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden_size, frame_hidden),
            nn.Dropout(p)
        )
        assert t2t_params is not None and n_vecs is not None
        tp = t2t_params.copy()
        self.fold = nn.Fold(**tp)
        del tp['output_size']
        self.unfold = nn.Unfold(**tp)
        self.n_vecs = n_vecs

    def forward(self, x, n_vecs=0, output_h=0, output_w=0):
        x = self.conv1(x)
        b, n, c = x.size()
        if n_vecs != 0:
            normalizer = x.new_ones(b, n, self.kernel_shape).view(-1, n_vecs, self.kernel_shape).permute(0, 2, 1)
            x = self.unfold(F.fold(x.view(-1, n_vecs, c).permute(0, 2, 1), output_size=(output_h, output_w),
                                   kernel_size=self.t2t_params['kernel_size'], stride=self.t2t_params['stride'],
                                   padding=self.t2t_params['padding']) / F.fold(normalizer,
                                                                                output_size=(output_h, output_w),
                                                                                kernel_size=self.t2t_params[
                                                                                    'kernel_size'],
                                                                                stride=self.t2t_params['stride'],
                                                                                padding=self.t2t_params[
                                                                                    'padding'])).permute(0,
                                                                                                         2,
                                                                                                         1).contiguous().view(
                b, n, c)
        else:
            normalizer = x.new_ones(b, n, self.kernel_shape).view(-1, self.n_vecs, self.kernel_shape).permute(0, 2, 1)
            x = self.unfold(self.fold(x.view(-1, self.n_vecs, c).permute(0, 2, 1)) / self.fold(normalizer)).permute(0,
                                                                                                                    2,
                                                                                                                    1).contiguous().view(
                b, n, c)
        x = self.conv2(x)
        return x


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


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)
