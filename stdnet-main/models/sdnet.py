"""
SDNet

Author: Zhuo Su
Date: March 16, 2023
"""

import math
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .std_utils import Conv2d, CSAM, InitialBlock, DiffBlock
from .ops import createConvFunc
from .config import config_model


class CDCM(nn.Module):
    """
    SDNet uses a slightly different version of CDCM compared with STDNet, so we define it here separately
    """
    def __init__(self, in_channels, out_channels, dils):
        super(CDCM, self).__init__()
        #print('dilation size: %s' % str(dils))

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv_dils = nn.ModuleList()
        for dil in dils:
            if dil == 1:
                self.conv_dils.append(nn.Sequential())
            else:
                self.conv_dils.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dil, padding=dil, bias=False))
        nn.init.constant_(self.conv1.bias, 0)
        
    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        y = self.conv_dils[0](x)
        for conv_dil in self.conv_dils[1:]:
            y = y + conv_dil(x)
        return y


class RefineLayer(nn.Module):
    """
    Reduce feature maps
    """
    def __init__(self, in_channels, out_channels):
        super(RefineLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class SDNet(nn.Module):
    def __init__(self, diffconvs, inplane=60, dil=24, bn=False, kernel_size=3):
        super(SDNet, self).__init__()
        self.fuseplanes = []

        self.dil = dil
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
        self.block2_3 = DiffBlock(diffconvs[6], self.inplane, self.inplane, bn=bn)
        self.block2_4 = DiffBlock(diffconvs[7], self.inplane, self.inplane, bn=bn, kernel_size=kernel_size)
        self.fuseplanes.append(self.inplane) # 2N
        
        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = DiffBlock(diffconvs[8], inplane, self.inplane, stride=2, bn=bn)
        self.block3_2 = DiffBlock(diffconvs[9], self.inplane, self.inplane, bn=bn)
        self.block3_3 = DiffBlock(diffconvs[10], self.inplane, self.inplane, bn=bn)
        self.block3_4 = DiffBlock(diffconvs[11], self.inplane, self.inplane, bn=bn, kernel_size=kernel_size)
        self.fuseplanes.append(self.inplane) # 4N

        self.block4_1 = DiffBlock(diffconvs[12], self.inplane, self.inplane, stride=2, bn=bn)
        self.block4_2 = DiffBlock(diffconvs[13], self.inplane, self.inplane, bn=bn)
        self.block4_3 = DiffBlock(diffconvs[14], self.inplane, self.inplane, bn=bn)
        self.block4_4 = DiffBlock(diffconvs[15], self.inplane, self.inplane, bn=bn, kernel_size=kernel_size)
        self.fuseplanes.append(self.inplane) # 4N

        dils_list = [
                [1, 3],
                [1, 3, 5],
                [3, 5, 7, 9],
                [5, 7, 9, 11],
                ]

        branch_channels = []
        self.attentions = nn.ModuleList()
        self.dilations = nn.ModuleList()
        for i in range(4):
            self.dilations.append(CDCM(self.fuseplanes[i], self.dil, dils=dils_list[i]))
            self.attentions.append(CSAM(self.dil))
            branch_channels.append(self.dil)
        self.conv_reduces = self.refines(branch_channels)

    def refines(self, c):
        conv_reduces = nn.ModuleList()
        conv_reduces.append(RefineLayer(c[0] + c[1], 1))
        for i in range(1, len(c) - 1):
            conv_reduces.append(RefineLayer(c[i] + c[i+1], c[i]))
        return conv_reduces

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

    def reparameterize(self):
        for layer in self.modules():
            if isinstance(layer, (InitialBlock, DiffBlock)):
                layer.reparameterize()

    def forward(self, x):
        x_shape = x.size()[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        x_fuses = []
        for i, xi in enumerate([x1, x2, x3, x4]):
            x_fuses.append(self.attentions[i](self.dilations[i](xi)))

        e4 = F.interpolate(x_fuses[3], x_fuses[2].shape[2:], mode="bilinear", align_corners=False) # dil

        e3 = self.conv_reduces[2](torch.cat([x_fuses[2], e4], dim=1)) # 2 * dil -> dil
        e3 = F.interpolate(e3, x_fuses[1].shape[2:], mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](torch.cat([x_fuses[1], e3], dim=1)) # 2 * dil -> dil
        e2 = F.interpolate(e2, x_fuses[0].shape[2:], mode="bilinear", align_corners=False)

        e1 = self.conv_reduces[0](torch.cat([x_fuses[0], e2], dim=1)) # 2 * dil -> 1
        e1 = F.interpolate(e1, x_shape, mode="bilinear", align_corners=False)
        output = torch.sigmoid(e1)

        return output


def sdnet(args):
    diffconvs = config_model(args.config)
    kernel_size = 5 if args.config == 'baseline' else 3 # to reparameterize RPDC
    return SDNet(diffconvs, bn=args.bn, kernel_size=kernel_size)

