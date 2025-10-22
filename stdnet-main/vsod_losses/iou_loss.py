"""
code modified from  https://github.com/Roudgers/DCFNet/blob/main/libs/utils/pytorch_iou/__init__.py

Author: Zhuo Su
Date: March 16, 2023
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def _iou(pred, target, mask, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        re_IoU1 = (1 - IoU1)
        if mask is not None:
            re_IoU1 = mask[i] * re_IoU1

        IoU = IoU + re_IoU1

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target, mask=None):

        return _iou(pred, target, mask, self.size_average)
