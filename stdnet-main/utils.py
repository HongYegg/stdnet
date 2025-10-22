"""
Utilities

Author: Zhuo Su
Date: March 16, 2023
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import shutil
import math
import time
import random
import skimage
import numpy as np
from skimage import io
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


######################################
#       measurement functions        #
######################################

def get_model_parm_nums(model):
    total = sum([param.numel() for param in model.parameters()])
    total = float(total) / 1e6
    return total



######################################
#         basic functions            #
######################################

def state_from_training(model_training_time, model_inference_time):
    training_dict = model_training_time.state_dict()
    update_dict = {}
    for k in model_inference_time.state_dict().keys():
        update_dict[k] = training_dict[k]
    model_inference_time.load_state_dict(update_dict)

def load_pretrained_model(model, path, running_file):

    def overlap(name, namelist):
        for n in namelist:
            if n in name:
                return n
        return None

    loadinfo = 'loading pretrained model'
    print(loadinfo)
    running_file.write('%s\n' % (loadinfo))
    running_file.flush()

    if path is None or not os.path.exists(path):
        raise ValueError('pretrained model not exists, given path: %s' % str(path))
    pretrained_state = torch.load(path, map_location='cpu')
    pretrained_state = pretrained_state['state_dict']
    state = model.state_dict()
    module_names = ['init_block', 'block1', 'block2', 'block3', 'block4']
    copynames = []
    for pname in state.keys():
        p = overlap(pname, module_names)
        if p is not None:
            pos = pname.index(p)
            for pname_pretrain in pretrained_state.keys():
                if pname[pos:] in pname_pretrain:
                    state[pname].data.copy_(pretrained_state[pname_pretrain].data)
                    copynames.append([pname, pname_pretrain])

    loadinfo = 'Loded pretrained model, totally loaded %d parameters' % len(copynames)
    print(loadinfo)
    running_file.write('%s: %s' % (loadinfo, str(copynames)))
    running_file.flush()

def finetune_checkpoint(model, path, running_file):
    loadinfo = 'loading pretrained model'
    print(loadinfo)
    running_file.write('%s\n' % (loadinfo))
    running_file.flush()

    if path is None or not os.path.exists(path):
        raise ValueError('pretrained model not exists, given path: %s' % str(path))
    pretrained_state = torch.load(path, map_location='cpu')
    pretrained_state = pretrained_state['state_dict']
    new_state = {}
    state = model.state_dict()

    for name, value in pretrained_state.items():
        if 'init_block.weight' in name:
            new_state[name.replace('init_block.', 'init_block.conv.0.')] = value
        elif 'conv1.' in name and 'block' in name:
            new_state[name.replace('conv1.', 'conv1.0.')] = value
        elif 'bn1.' in name and 'block' in name:
            new_state[name.replace('bn1.', 'bn1.0.')] = value
        else:
            new_state[name] = value

    state.update(new_state)
    model.load_state_dict(state)
    print('successfully preloaded checkpoint from %s' % path)

def load_checkpoint(args, running_file=None):

    model_dir = os.path.join(args.savedir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = ''

    if args.evaluate == 'best':
        model_filename = os.path.join(model_dir, 'checkpoint_best.pth')
    elif args.evaluate is not None:
        model_filename = args.evaluate
    elif args.resume_from is not None:
        model_filename = args.resume_from
    else:
        if os.path.exists(latest_filename):
            with open(latest_filename, 'r') as fin:
                model_filename = fin.readlines()[0].strip()
    loadinfo = "=> loading checkpoint from '{}'".format(model_filename)
    print(loadinfo)

    state = None
    if os.path.exists(model_filename):
        state = torch.load(model_filename, map_location='cpu')
        loadinfo2 = "=> loaded checkpoint '{}' successfully".format(model_filename)
    else:
        loadinfo2 = "no checkpoint loaded"
    print(loadinfo2)
    if running_file is not None:
        running_file.write('%s\n%s\n' % (loadinfo, loadinfo2))
        running_file.flush()

    return state

def preload_stage1_weights(model, state):

    model_state = model.state_dict()
    for k, v in model_state.items():
        if 'block' in k:
            model_state[k] = state[k]
    model.load_state_dict(model_state)


def save_checkpoint(state, epoch, root, saveID, is_best, keep_freq=10):

    filename = 'checkpoint_%03d.pth' % epoch
    model_dir = os.path.join(root, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # write new checkpoint 
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    print("=> saved checkpoint '{}'".format(model_filename))

    # update best model 
    if isinstance(is_best, dict):
        for k, v in is_best.items():
            if v:
                best_filename = os.path.join(model_dir, 'checkpoint_best_{}.pth'.format(k))
                shutil.copyfile(model_filename, best_filename)
    elif is_best:
        best_filename = os.path.join(model_dir, 'checkpoint_best.pth')
        shutil.copyfile(model_filename, best_filename)

    # remove old model
    if saveID is not None and (saveID + 1) % keep_freq != 0:
        filename = 'checkpoint_%03d.pth' % saveID
        model_filename = os.path.join(model_dir, filename)
        if os.path.exists(model_filename):
            os.remove(model_filename)
            print('=> removed checkpoint %s' % model_filename)

    print('##########Time##########', time.strftime('%Y-%m-%d %H:%M:%S'))
    return epoch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        #self.sum += val * n
        self.sum += val
        self.count += n
        self.val = val / n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args):
    """
    pretrained:
        groupname = ['conv weigths', 'norm weights', 'relu weights', 'block weights']
    scratch:
        groupname = ['conv weigths', 'norm weights', 'relu weights']
    scratch with scheduling scales:
        groupname = ['conv weigths', 'norm weights', 'relu weights', 'scale weights']

    """

    method = args.lr_type
    if method == 'cosine':
        T_total = float(args.epochs)
        T_cur = float(epoch)
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        lr = args.lr
        for epoch_step in args.lr_steps:
            if epoch >= epoch_step:
                lr = lr * 0.1

    lr_factors = args.lr_factors
    lr_str = []
    ## adjust lr for normal weights
    for param_group, factor in zip(optimizer.param_groups, lr_factors):
        param_group['lr'] = lr * factor
        lr_str.append(lr * factor)

    ## if there are scale weights to tune
    if hasattr(args, 'scale_id') and epoch < args.warmup:
        optimizer.param_groups[args.scale_id]['lr'] = 0.
        lr_str[args.scale_id] = 0.

    lr_str = '-'.join(['{:.6f}'.format(l) for l in lr_str])
    return lr_str


######################################
#     edge specific functions        #
######################################


def cross_entropy_loss_RCF(prediction, labelf, beta):
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label==1).float()
    num_negative = torch.sum(label==0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    cost = F.binary_cross_entropy(
            prediction, labelf, weight=mask, reduction='sum')

    return cost

def dice_loss(prediction, labelf):

    dist = (torch.sum(prediction ** 2) + torch.sum(labelf ** 2)) / (2 * torch.sum(prediction * labelf) + 1)

    return dist

def l1_loss(weights, l1wd):
    loss = 0
    for w in weights:
        loss += torch.sum(w.abs())

    return l1wd * loss

######################################
#     vsod specific functions        #
######################################

from vsod_losses import IOU, SSIM

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = SSIM(window_size=11,size_average=True)
iou_loss = IOU(size_average=True)

def bce_ssim_loss(pred, target, mask=None):

    #bce_out = bce_loss(pred, target)
    weight = None
    if mask is not None:
        if mask.ndim < target.ndim:
            weight = mask.view(-1, 1, 1, 1).expand_as(target)
            mask = mask.view(-1,)
        else:
            weight = mask.view(target.shape)
            mask = None
    bce_out = F.binary_cross_entropy(pred, target, weight=weight, reduction='mean')
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target, mask=mask)

    loss = bce_out + ssim_out + iou_out

    return loss, bce_out.item(), ssim_out.item(), iou_out.item()


######################################
#      sod specific functions        #
######################################


def cross_entropy_loss_sod(prediction, labelf, mask=None):
    cost = F.binary_cross_entropy(
            prediction, labelf, weight=mask, reduction='sum')

    return cost

# loss function: seven probability map --- 6 scale + 1 fuse
class LossDSS(nn.Module):
    """
    for DSS
    """
    def __init__(self, weight=[1.0] * 7):
        super(LossDSS, self).__init__()
        self.weight = weight

    def forward(self, x_list, label, mask=None):
        loss = self.weight[0] * F.binary_cross_entropy(x_list[0], label)
        for i, x in enumerate(x_list[1:]):
            loss += self.weight[i + 1] * F.binary_cross_entropy(x, label)
        return loss * x_list[0].shape[0]

class CrossEntropyLossFal(nn.Module):
    """
    This loss is for "SAMNet: Stereoscopically Attentive Multi-scale Network for Lightweight Salient Object Detection" (IEEE TIP) and "Lightweight Salient Object Detection via Hierarchical Visual Perception Learning" (IEEE TCYB).
    """
    def __init__(self):
        super(CrossEntropyLossFal, self).__init__()

    def forward(self, inputs, target, mask=None):
        if isinstance(target, tuple):
            target = target[0]
        target = target.float()
        if target.ndim==4:
            target = target.squeeze(1)
        loss = F.binary_cross_entropy(inputs[:, 0, :, :], target)
        for i in range(1, inputs.shape[1]):
            loss += 0.4 * F.binary_cross_entropy(inputs[:, i, :, :], target)

        return loss * inputs.shape[0]

class CrossEntropyLossICON(nn.Module):
    """
    This loss is for ICON
    """
    def __init__(self):
        super(CrossEntropyLossICON, self).__init__()

    def forward(self, inputs, target, mask=None):
        loss = F.binary_cross_entropy(inputs[0], target)
        for pred in inputs[1:]:
            loss = loss + F.binary_cross_entropy(pred, target)

        return loss * inputs[0].shape[0]

def BCEDiceLoss(inputs, targets):
    if targets.ndim == 4:
        targets = targets.squeeze(1)
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    #print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return (bce + 1 - dice) * inputs.shape[0]

class DSBCEDiceLoss(nn.Module):
    """
    This loss is for EDN: "Edn: Salient object detection via extremelydownsampled network, TIP, 2022"
    """
    def __init__(self):
        super(DSBCEDiceLoss, self).__init__()

    def forward(self, inputs, target, mask=None):
        #pred1, pred2, pred3, pred4, pred5 = tuple(inputs)
        if isinstance(target, tuple):
           target = target[0]
        #target = target[:,0,:,:]
        loss1 = BCEDiceLoss(inputs[:,0,:,:], target)
        loss2 = BCEDiceLoss(inputs[:,1,:,:], target)
        loss3 = BCEDiceLoss(inputs[:,2,:,:], target)
        loss4 = BCEDiceLoss(inputs[:,3,:,:], target)
        loss5 = BCEDiceLoss(inputs[:,4,:,:], target)
        
        return loss1+loss2+loss3+loss4+loss5

######################################
#         debug functions            #
######################################

# no function currently
