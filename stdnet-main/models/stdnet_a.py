"""
STDNet-A

Author: Zhuo Su
Date: March 16, 2023
"""

import math
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import config_model
from .std_utils import (
        Conv2d,
        ProjectLayer,
        STDM,
        CDCM,
        CSAM,
        InitialBlock,
        DiffBlock,
    )
from .attention_block import MobileVitV2Block
from .ops import createConvFunc


class STDNetA(nn.Module):
    def __init__(self, diffconvs, tid, scales, inplane=60, dil=24, nframes=1, nstdc=1, bn=False, kernel_size=3):
        super(STDNetA, self).__init__()
        assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil
        self.nframes = nframes
        self.nstdc = nstdc
        self.fuseplanes = []

        self.inplane = inplane
        if isinstance(diffconvs[0], list):
            self.init_block = InitialBlock(diffconvs[0], 3, self.inplane)
        else:
            self.init_block = Conv2d(createConvFunc(diffconvs[0]), 3, self.inplane, kernel_size=3, padding=1)

        self.block1_1 = DiffBlock(diffconvs[1], self.inplane, self.inplane, stride=2, bn=bn)
        self.block1_2 = DiffBlock(diffconvs[2], self.inplane, self.inplane, bn=bn)
        self.block1_3 = DiffBlock(diffconvs[3], self.inplane, self.inplane, bn=bn, kernel_size=kernel_size)
        self.fuseplanes.append(self.inplane) # N

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = DiffBlock(diffconvs[4], inplane, self.inplane, stride=2, bn=bn)
        self.block2_2 = DiffBlock(diffconvs[5], self.inplane, self.inplane, bn=bn)
        self.block2_3 = MobileVitV2Block(self.inplane, self.inplane, 2, 2)
        self.fuseplanes.append(self.inplane) # 2N
        
        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = DiffBlock(diffconvs[6], inplane, self.inplane, stride=2, bn=bn)
        self.block3_2 = DiffBlock(diffconvs[7], self.inplane, self.inplane, bn=bn)
        self.block3_3 = MobileVitV2Block(self.inplane, self.inplane, 2, 2)
        self.fuseplanes.append(self.inplane) # 4N

        self.block4_1 = DiffBlock(diffconvs[8], self.inplane, self.inplane, stride=2, bn=bn)
        self.block4_2 = DiffBlock(diffconvs[9], self.inplane, self.inplane, bn=bn)
        self.block4_3 = MobileVitV2Block(self.inplane, self.inplane, 2, 1)
        self.fuseplanes.append(self.inplane) # 4N

        self.project1 = nn.ModuleList()
        self.global_spatial = nn.ModuleList()
        self.global_temporal = nn.ModuleList() if nframes > 1 else None
        self.project2 = nn.ModuleList()
        self.attention = nn.ModuleList()

        for inplane, outplane in zip(self.fuseplanes, [dil, dil, dil, 2 * dil]):
            self.project1.append(ProjectLayer(inplane, outplane))
            self.global_spatial.append(CDCM(2 * dil, dil))
            if nframes > 1:
                self.global_temporal.append(STDM(2 * dil, dil, nframes, tid, scales, nstdc, bn=bn))
            self.project2.append(ProjectLayer(2 * dil, dil, relu=False))
            self.attention.append(CSAM(self.dil))
        self.project_final = ProjectLayer(self.dil, 1, relu=False, bn=False)

    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        scale_weights = []
        for pname, p in self.named_parameters():
            if 'scales' in pname:
                scale_weights.append(p)
            elif 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)
        return conv_weights, bn_weights, relu_weights, scale_weights

    def get_stage2_weights(self):
        conv_weights = []
        norm_weights = []
        relu_weights = []
        block_weights = []
        for pname, p in self.named_parameters():
            if 'block' not in pname:
                if 'norm' in pname:
                    norm_weights.append(p)
                elif 'relu' in pname:
                    relu_weights.append(p)
                else:
                    conv_weights.append(p)
            else:
                block_weights.append(p)
        return conv_weights, norm_weights, relu_weights, block_weights

    def reparameterize(self):
        for layer in self.modules():
            if isinstance(layer, (InitialBlock, DiffBlock, STDM)):
                layer.reparameterize()

    def forward(self, x):
        """
        nframes=1: forward on images, not implemented
        nframes>1: forward on video clip
        """
        b, _, h, w = x.size() # batchsize, nframes * 3, h, w
        x_shape = x.size()[2:]

        if self.nframes > 1:
            x = x.view(b * self.nframes, 3, h, w)

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)

        x_pre = None
        for _i, xi in enumerate([x4, x3, x2, x1]):
            i = 3 - _i
            xi = self.project1[i](xi) # dil (for i = 0, 1, 2) or 2*dil (for i = 3)
            if x_pre is not None:
                x_pre = F.interpolate(x_pre, xi.shape[2:], mode="bilinear", align_corners=False)
                xi = torch.cat([xi, x_pre], dim=1) # 2*dil
            xi_spatial = self.global_spatial[i](xi) # 2*dil -> dil
            if self.global_temporal is None:
                xi_temporal = xi_spatial
            else:
                xi_temporal = self.global_temporal[i](xi) # 2*dil -> dil
            xi_spatemp = torch.cat([xi_spatial, xi_temporal], dim=1) # 2*dil
            xi = self.project2[i](xi_spatemp) # 2*dil -> dil
            x_pre = self.attention[i](xi) # dil

        output = self.project_final(x_pre)
        output = F.interpolate(output, x_shape, mode="bilinear", align_corners=False)
        output = torch.sigmoid(output)
        return output


def stdneta(args):
    """
    tid: temporal ops like cv, cd, ad.
    scales: scaling factors attached to ops.
    """
    diffconvs = config_model(args.config)
    nframes = args.nframes if hasattr(args, 'nframes') else 1
    kernel_size = 5 if args.config == 'baseline' else 3 # to reparameterize RPDC
    return STDNetA(diffconvs, tid=args.tid, scales=args.scales, nframes=nframes, bn=args.bn, kernel_size=kernel_size)
