"""
Testing FPS of SOD models

Author: Zhuo Su
Date: March 16, 2023
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import numpy as np
import time
import models
from utils import get_model_parm_nums

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='Speed of SOD models')

parser.add_argument('--model', type=str, default='sdnet', 
        help='model to train the dataset')
parser.add_argument('--config', type=str, default='baseline', 
        help='model configurations, please refer to models/config.py for possible configurations')
parser.add_argument('--gpu', type=str, default='', 
        help='gpus available')
parser.add_argument('--size', type=int, default=None, 
        help='size of samples')

parser.add_argument('-j', '--workers', type=int, default=4, 
        help='number of data loading workers')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def main():

    global args

    ### Refine args
    args.bn = False
    args.use_cuda = torch.cuda.is_available()
    #args.use_cuda = False

    if args.size is not None:
        args.size = (args.size, args.size)

    print(args)

    ### Create model
    model = getattr(models, args.model)(args)

    count_paramsM = get_model_parm_nums(model)
    print('Model size: %f MB' % count_paramsM)

    ### Transfer to cuda devices
    if args.use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        print('cuda is used, with %d gpu devices' % torch.cuda.device_count())
    else:
        print('cuda is not used, the running might be slow')

    cudnn.benchmark = True
    computefps(model, args)

def computefps(model, args):
    input = torch.randn(1, 3, args.size[0], args.size[1])
    if args.use_cuda:
        input = input.cuda()

    model.eval()

    time_spent = []
    if args.use_cuda:
        torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
    n = 1000 if args.use_cuda else 110
    nbase = int(n * 0.1)
    for idx in range(n):
        start_time = time.time()
        with torch.no_grad():
            _ = model(input)

        if args.use_cuda:
            torch.cuda.synchronize()
        if idx > nbase:
            time_spent.append(time.time() - start_time)
    print('Average speed: {:.4f} fps'.format(1.0 / np.mean(time_spent)))


if __name__ == '__main__':
    main()
    print('done')
