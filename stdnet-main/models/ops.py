"""
Function factory for diff convolutional operations.
Modified from PiDiNet, iccv21: "https://github.com/zhuoinoulu/pidinet"

Author: Zhuo Su
Date: March 16, 2023
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

## sptial cd, ad, rd convolutions
def createConvFunc(op_type, theta=1.0):
    assert op_type in ['cv', 'cd', 'ad', 'rd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return F.conv2d

    assert theta > 0 and theta <= 1.0, 'theta should be within (0, 1]'

    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True) * theta
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - theta * weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape) # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:] * theta
            buffer[:, :, 12] = weights[:, :, 0] * (1 - theta)
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    else:
        print('impossible to be here unless you force that')
        return None


## tempral cd, ad convolutions
def createTConvFunc(op_type, theta=1.0):
    assert op_type in ['cv', 'cd', 'ad'], 'unknown op type: %s' % str(op_type)
    """
    shape of x: b, c, t, h (or w)
    """
    if op_type == 'cv':
        def func(x, weights, bias=None, groups=1):
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            x = F.pad(x, (0, 0, 1, 1), mode='replicate') # b, c, t+2, h
            return F.conv2d(x, weights, bias, padding=(0, 1), groups=groups)
        return func
    elif op_type == 'cd':
        def func(x, weights, bias=None, groups=1):
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'

            weights_c = weights.sum(dim=[2, 3], keepdim=True) * theta
            x_pad = F.pad(x, (0, 0, 1, 1), mode='replicate') # b, c, t+2, h
            yc = F.conv2d(x, weights_c, groups=groups)
            y = F.conv2d(x_pad, weights, bias, padding=(0, 1), groups=groups)
            return y - yc
        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, groups=1):
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - theta * weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape) # clock-wise
            x = F.pad(x, (0, 0, 1, 1), mode='replicate') # b, c, t+2, h
            y = F.conv2d(x, weights_conv, bias, padding=(0, 1), groups=groups)
            return y
        return func


## reparameterize PDC/STDC functions to standard convolution
def reparameterize(weight, op_type, use_rd=False):
    assert op_type in ['cv', 'cd', 'ad', 'rd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'rd':
        shape = weight.shape
        buffer = torch.zeros(shape[0], shape[1], 5 * 5, device=weight.device)
        weight = weight.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weight[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weight[:, :, 1:]
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        return buffer
    if op_type == 'cd':
        shape = weight.shape
        weight_c = weight.sum(dim=[2, 3])
        weight = weight.view(shape[0], shape[1], -1)
        weight[:, :, 4] = weight[:, :, 4] - weight_c
        weight = weight.view(shape)
    elif op_type == 'ad':
        shape = weight.shape
        weight = weight.view(shape[0], shape[1], -1)
        weight = (weight - weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)
    if use_rd:
        shape = weight.shape
        weight = weight.view(shape[0], shape[1], -1)
        buffer = torch.zeros(shape[0], shape[1], 5 * 5, device=weight.device)
        buffer[:, :, [6, 7, 8, 11, 12, 13, 16, 17, 18]] = weight[:, :, :]
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        return buffer
    else:
        return weight

