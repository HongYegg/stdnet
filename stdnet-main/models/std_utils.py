"""
Modules of SDNet/SDNet-A/STDNet/STDNet-A

Author: Zhuo Su
Date: March 16, 2023
"""

import math
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import createConvFunc, createTConvFunc, reparameterize
from .config import config_model

#############################################
## Spatial and Temporal Difference Modules ##
#############################################

class Conv2d(nn.Module):
    def __init__(self, diffconv, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.diffconv = diffconv

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        return self.diffconv(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class TConv2d(nn.Module):
    def __init__(self, diffconv, in_channels, out_channels, groups=1, bias=False):
        super(TConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, 3, 3))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.diffconv = diffconv

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):

        return self.diffconv(input, self.weight, self.bias, self.groups)


class STDC(nn.Module):
    """
    SpatioTemporal Difference Convolution
    """
    def __init__(self, in_channels, tid, scales, bn=True):
        super(STDC, self).__init__()
        assert len(tid) == len(scales), 'length of tid should match the length of scales'
        self.bn = bn
        self.tid = tid

        if len(tid) > 1:
            scales = [float(v) for v in scales]
            self.scales = nn.Parameter(torch.zeros(len(scales),))
            self.scales.data.copy_(torch.tensor(scales))

            self.convs = nn.ModuleList()
            if self.bn: self.norms = nn.ModuleList()
            for op in tid:
                func = createTConvFunc(op)
                self.convs.append(TConv2d(func, in_channels, in_channels))
                if self.bn: self.norms.append(nn.BatchNorm2d(in_channels))
        else:
            func = createTConvFunc(tid[0])
            self.convs = TConv2d(func, in_channels, in_channels, bias=False if self.bn else True)
            if self.bn: self.norms = nn.BatchNorm2d(in_channels)

    def reparameterize(self):
        assert len(self.tid) > 1, 'diffconv should be a list longer than 1 for DCR'
        assert self.bn, 'bn should be True for DCR'
        rep_weight = False
        scales = torch.softmax(self.scales, dim=0).detach()
        for i in range(len(self.tid)):
            w = reparameterize(self.convs[i].weight.data, self.tid[i], use_rd=False) 

            mu = self.norms[i].running_mean
            sigma = torch.sqrt(self.norms[i].running_var + self.norms[i].eps)
            gamma = self.norms[i].weight
            beta = self.norms[i].bias
            w = w * gamma.view(-1, 1, 1, 1) / sigma.view(-1, 1, 1, 1)
            bias = beta - mu * gamma / sigma
            w = w * scales.data[i] 
            bias = bias * scales.data[i]
            if not rep_weight:
                self.convs.register_buffer('weight', w)
                self.convs.register_buffer('bias', bias)
                rep_weight = True
            else:
                self.convs.weight.add_(w)
                self.convs.bias.add_(bias)

    def forward(self, x):
        out = []
        if len(self.tid) > 1:
            scales = torch.softmax(self.scales, dim=0)
            for i in range(len(self.convs)):
                out_i = self.convs[i](x)
                if self.bn: out_i = self.norms[i](out_i)
                out_i = scales[i] * out_i
                out.append(out_i)
            y = sum(out)
        else:
            y = self.convs(x)
            if self.bn: y = self.norms(y)
        return y

class STDCBase(nn.Module):
    def __init__(self, channels, tid, scales):
        super(STDCBase, self).__init__()
        self.tdc1 = STDC(channels, tid, scales)
        self.tdc2 = STDC(channels, tid, scales)
        self.relu = nn.ReLU()

    def forward(self, x):
        ## shape of x: b, h, c, t, w
        b, h, c, t, w = x.shape
        x = x.view(b * h, c, t, w)
        x = x + self.relu(self.tdc1(x)) # w-t plane TDC

        x = x.view(b, h, c, t, w).transpose(1, 4).contiguous().view(b * w, c, t, h) # b*w, c, t, h
        x = x + self.relu(self.tdc2(x)) # h-t plane TDC

        x = x.view(b, w, c, t, h).transpose(1, 4).contiguous()

        return x # b, h, c, t, w


#############################################
##      Basic SDNet and STDNet modules     ##
#############################################

class ProjectLayer(nn.Module):
    """
    Reduce feature maps
    """
    def __init__(self, in_channels, out_channels, bn=True, relu=True):
        super(ProjectLayer, self).__init__()
        bias = False if bn else True
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)
        if bias:
            nn.init.constant_(self.conv.bias, 0)
        self.norm = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.relu(x) if self.relu is not None else x
        return x


class CDCM(nn.Module):
    def __init__(self, in_channels, out_channels, dils=[1, 3, 5]):
        super(CDCM, self).__init__()
        #print('dilation size: %s' % str(dils))

        self.conv1 = ProjectLayer(in_channels, out_channels, bn=False, relu=False)
        self.conv_dils = nn.ModuleList()
        for dil in dils:
            if dil == 1:
                self.conv_dils.append(nn.Sequential())
            else:
                self.conv_dils.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dil, padding=dil, bias=False))

        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        y = self.conv_dils[0](x)
        for conv_dil in self.conv_dils[1:]:
            y = y + conv_dil(x)
        y = self.norm(y)
        y = self.relu(y)
        return y


class STDM(nn.Module):
    """
    Temporal transformer
    input: b*t, c, h, w
    """
    def __init__(self, in_channels, out_channels, nframes, tid, scales, nstdc, bn=True):
        super(STDM, self).__init__()
        self.nframes = nframes
        self.tid = tid
        self.nstdc = nstdc
        if nstdc == 1:
            self.tdc1 = STDC(in_channels, tid, scales, bn=bn)
            self.tdc2 = STDC(in_channels, tid, scales, bn=bn)
            self.relu = nn.ReLU()
        else:
            tdc_layers = []
            for i in range(nstdc):
                tdc_layers.append(STDCBase(in_channels, tid, scales))
            self.tdc = nn.Sequential(*tdc_layers)

        self.project2 = ProjectLayer(in_channels, out_channels)

    def reparameterize(self):
        assert self.nstdc == 1, 'now, only reparameterize model with nstdc=1. can be easily done for nstdc>1'
        self.tdc1.reparameterize()
        self.tdc2.reparameterize()


    def forward_1(self, x):
        ## shape of x: b, h, c, t, w
        b, h, c, t, w = x.shape
        x = x.view(b * h, c, t, w)
        x = x + self.relu(self.tdc1(x)) # w-t plane TDC

        x = x.view(b, h, c, t, w).transpose(1, 4).contiguous().view(b * w, c, t, h) # b*w, c, t, h
        x = x + self.relu(self.tdc2(x)) # h-t plane TDC

        x = x.view(b, w, c, t, h).permute(0, 3, 2, 4, 1).contiguous().view(b * t, c, h, w)

        return x

    def forward(self, x):
        bt, c, h, w = x.size() # shape of x: b*t, c, h, w
        t = self.nframes
        b = bt // t
        x = x.view(b, t, c, h, w).transpose(1, 3).contiguous() # b, h, c, t, w
        if self.nstdc == 1:
            x = self.forward_1(x)
        else:
            x = self.tdc(x) # b, h, c, t, w
            x = x.transpose(1, 3).contiguous().view(b * t, c, h, w)
        x = self.project2(x) # c -> c_out
        return x


class CSAM(nn.Module):
    """
    Sample-adaptive spatial attention
    """
    def __init__(self, channels):
        super(CSAM, self).__init__()

        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        return x * y


class InitialBlock(nn.Module):
    def __init__(self, diffconv, inplane, ouplane):
        super(InitialBlock, self).__init__()
        self.diffconv = diffconv
        self.conv = nn.ModuleList()
        for diffconv_i in diffconv:
            _func = createConvFunc(diffconv_i)
            self.conv.append(Conv2d(_func, inplane, ouplane, kernel_size=3, padding=1))

        n_conv = len(self.conv)
        self.scales = nn.Parameter(torch.ones(n_conv,))
        self.scales.data[0].copy_(torch.tensor(0.))

    def reparameterize(self):
        rep_weight = False
        scales = torch.softmax(self.scales, dim=0).detach()
        for i in range(len(self.diffconv)):
            w = scales.data[i] * reparameterize(self.conv[i].weight.data, self.diffconv[i]) 
            if not rep_weight:
                self.register_buffer('weight', w)
                rep_weight = True
            else:
                self.weight.add_(w)

    def forward(self, x):
        out = []
        scales = torch.softmax(self.scales, dim=0)
        for i in range(len(self.conv)):
            out.append(scales[i] * self.conv[i](x))
        y = sum(out)
        return y


class DiffBlock(nn.Module):
    def __init__(self, diffconv, inplane, ouplane, stride=1, bn=True, kernel_size=3):
        super(DiffBlock, self).__init__()
        self.stride=stride
        self.bn = bn
        self.difflist = False
        #print('use bn? %s' % (str(self.bn)))
        padding = kernel_size // 2
        self.diffconv = diffconv
        self.inplane = inplane

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        if self.bn: self.bn0 = nn.BatchNorm2d(inplane)

        if isinstance(diffconv, list):
            self.difflist = True
            self.conv1 = nn.ModuleList()
            if self.bn: self.bn1 = nn.ModuleList()
            self.scales = nn.Parameter(torch.ones(len(diffconv),))
            self.scales.data[0].copy_(torch.tensor(0.))
            for diffconv_i in diffconv:
                _func = createConvFunc(diffconv_i)
                self.conv1.append(Conv2d(_func, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False))
                if self.bn: self.bn1.append(nn.BatchNorm2d(inplane)) # notice: affine=False
        else:
            _func = createConvFunc(diffconv)
            self.register_buffer('bias', torch.zeros(1, inplane, 1, 1))
            self.conv1 = Conv2d(_func, inplane, inplane, kernel_size=kernel_size, padding=padding, groups=inplane, bias=False if self.bn else True)
            if self.bn: self.bn1 = nn.BatchNorm2d(inplane) # notice: affine=False
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=True)
        self.relu1 = nn.ReLU()

    def reparameterize(self):
        assert isinstance(self.diffconv, list), 'diffconv should be a list for DCR'
        rep_weight = False
        scales = torch.softmax(self.scales, dim=0).detach()
        use_rd = 'rd' in self.diffconv
        self.register_buffer('bias', torch.zeros(1, self.inplane, 1, 1))
        for i in range(len(self.diffconv)):
            w = reparameterize(self.conv1[i].weight.data, self.diffconv[i], use_rd=use_rd) 
            if self.bn:
                mu = self.bn1[i].running_mean
                sigma = torch.sqrt(self.bn1[i].running_var + self.bn1[i].eps)
                gamma = self.bn1[i].weight
                beta = self.bn1[i].bias
                w = w * gamma.view(-1, 1, 1, 1) / sigma.view(-1, 1, 1, 1)
                bias = beta - mu * gamma / sigma
                w = w * scales.data[i] 
                bias = bias * scales.data[i]
                if not rep_weight:
                    self.conv1.register_buffer('weight', w)
                    self.conv1.register_buffer('bias', bias)
                    rep_weight = True
                else:
                    self.conv1.weight.add_(w)
                    self.conv1.bias.add_(bias)
            else:
                w = w * scales.data[i] 
                if not rep_weight:
                    self.conv1.register_buffer('weight', w)
                    self.conv1.register_buffer('bias', torch.zeros(self.inplane))
                    rep_weight = True
                else:
                    self.conv1.weight.add_(w)

        if self.bn:
            mu = self.bn0.running_mean
            sigma = torch.sqrt(self.bn0.running_var + self.bn0.eps)
            gamma = self.bn0.weight
            beta = self.bn0.bias

            a = (gamma / sigma).view(-1, 1, 1, 1)
            self.conv1.weight.mul_(a)

            bias = (sigma * beta / gamma - mu).view(1, -1, 1, 1)
            self.bias.copy_(bias)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = x
        if self.bn: y = self.bn0(y)
        if hasattr(self, 'bias'): y = y + self.bias
        if self.difflist:
            out = []
            scales = torch.softmax(self.scales, dim=0)
            for i in range(len(self.conv1)):
                out_i = self.conv1[i](y)
                if self.bn: out_i = self.bn1[i](out_i)
                out_i = scales[i] * out_i
                out.append(out_i)
            y = sum(out)
        else:
            y = self.conv1(y)
            if self.bn: y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

