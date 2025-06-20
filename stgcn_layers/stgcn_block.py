import torch
import numpy as np
import torch.nn as nn
import pdb
import math
import copy

class GCN_unit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        A,
        adaptive=True,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        assert A.size(0) == self.kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )
        self.adaptive = adaptive
        # print(self.adaptive)
        if self.adaptive:
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, len_x):
        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A)).contiguous()
        y = self.bn(x)
        y = self.relu(y)
        return y

class STGCN_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        A,
        adaptive=True,
        stride=1,
        dropout=0,
        residual=True,
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = GCN_unit(
            in_channels,
            out_channels,
            kernel_size[1],
            A,
            adaptive=adaptive,
        )
        if kernel_size[0] > 1:
            self.tcn = nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    (kernel_size[0], 1),
                    (stride, 1),
                    padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            )
        else:
            self.tcn = nn.Identity()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, len_x=None):
        res = self.residual(x)
        x = self.gcn(x, len_x)
        x = self.tcn(x) + res
        return self.relu(x)

class STGCNChain(nn.Sequential):
    def __init__(self, in_dim, block_args, kernel_size, A, adaptive):
        super(STGCNChain, self).__init__()
        last_dim = in_dim
        for i, [channel, depth] in enumerate(block_args):
            for j in range(depth):
                self.add_module(f'layer{i}_{j}', STGCN_block(last_dim, channel, kernel_size, A.clone(), adaptive))
                last_dim = channel

def get_stgcn_chain(in_dim, level, kernel_size, A, adaptive):
    if level == 'spatial':
        block_args = [[64,1], [128,1], [256,1]]
    elif level == 'temporal':
        block_args = [[256,3]]
    else:
        raise NotImplementedError
    return STGCNChain(in_dim, block_args, kernel_size, A, adaptive), block_args[-1][0]