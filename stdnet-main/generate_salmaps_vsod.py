"""
Generating saliency maps of VSOD models

Author: Zhuo Su
Date: March 16, 2023
"""


from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import time
import numpy as np
import models
from utils import state_from_training
from dataloader_vsod import prepare_test_data

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='Generate sal maps by STDNet')

parser.add_argument('--savedir', type=str, default='results/savedir', 
        help='path to save result and checkpoint')
parser.add_argument('--datadir', type=str, default='./dataset', 
        help='dir to the dataset')
parser.add_argument('--test-lst', type=str, default='dataset/test_video_lst.txt', 
        help='name of training dataset')
parser.add_argument('--testdata', type=str, default='DAVIS', 
        help='name of validation dataset')

parser.add_argument('--model', type=str, default='stdnet', 
        help='model to train the dataset')
parser.add_argument('--nobn', action='store_true',
        help='whether use bn in backbone')
parser.add_argument('--nframes', type=int, default=8, 
        help='number of frames in each input clip')
parser.add_argument('--tid', nargs='+', default=['cv'],
        help='temporal operations, in [cv, cd, ad]')
parser.add_argument('--scales', nargs='+', default=[1],
        help='scaling factors for temporal operations')
parser.add_argument('--framegap', type=int, default=4, 
        help='gap between two frames in the input')
parser.add_argument('--train-config', type=str, default='sdnet', 
        help='model configurations during training, please refer to models/config.py for possible configurations')
parser.add_argument('--inference-config', type=str, default='baseline', 
        help='model configurations during inference, please refer to models/config.py for possible configurations')
parser.add_argument('--seed', type=int, default=None, 
        help='random seed (default: None)')
parser.add_argument('--gpu', type=str, default='', 
        help='gpus available')

parser.add_argument('--size', type=int, default=None, 
        help='size of samples')
parser.add_argument('-j', '--workers', type=int, default=4, 
        help='number of data loading workers')

parser.add_argument('--evaluate', type=str, default=None, 
        help="full path to checkpoint to be evaluated or 'best'")

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def main():

    global args, best_mae

    ### Refine args
    args.use_cuda = torch.cuda.is_available()
    args.scales = [1.0 for _ in range(len(args.tid))]
    args.bn = not args.nobn

    if args.size is not None:
        args.size = (args.size, args.size)

    print(args)

    ### Create model
    args.config = args.train_config
    model_training_time = getattr(models, args.model)(args)

    args.config = args.inference_config
    args.bn = False
    args.tid = ['cv']
    args.scales = [1.0]
    model_inference_time = getattr(models, args.model)(args)

    ### Transfer to cuda devices
    if args.use_cuda:
        model_inference_time = torch.nn.DataParallel(model_inference_time).cuda()
        model_training_time = torch.nn.DataParallel(model_training_time).cuda()
        print('cuda is used, with %d gpu devices' % torch.cuda.device_count())
    else:
        print('cuda is not used, the running might be slow')

    #cudnn.benchmark = True

    print("=> loading checkpoint from '{}'".format(args.evaluate))
    checkpoint_dict = torch.load(args.evaluate, map_location='cpu')
    model_training_time.load_state_dict(checkpoint_dict)
    print("=> loaded checkpoint '{}' successfully; begin to reparameterize".format(args.evaluate))

    ### Difference Convolution Reparameterization (DCR)
    model_training_time.module.reparameterize()
    state_from_training(model_training_time, model_inference_time)
    print("=> reparameterization done")

    ### Generating saliency maps using reparameterized model
    test(model_inference_time, args)
    return

def test(model, args):

    from PIL import Image
    import scipy.io as sio
    model.eval()

    pth = os.path.basename(args.evaluate).split('.')[0].split('_')[2]
    save_dir = args.savedir

    ### Load Data
    testdatas = args.testdata.split('+')
    for tdat in testdatas:

        if tdat in ['DAVSOD', 'DAVIS']:
            folder_index = -3
        elif tdat in ['VOS']:
            folder_index = -2
        test_loader = prepare_test_data(args, dataset=tdat)

        img_dir = os.path.join(save_dir, '{}'.format(tdat))
        eval_info = '\nBegin to generating saliency maps for model {}...\nImg generated in {}\n'.format(args.model, img_dir)
        print(eval_info)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        else:
            print('%s already exits, but it will be regenerated' % img_dir)

        for idx, (image, img_names, imgsize) in enumerate(test_loader):

            imgsize = [x.item() for x in imgsize]
            with torch.no_grad():
                image = image.cuda() if args.use_cuda else image
                result = model(image)
                result = F.interpolate(result, mode='bilinear', 
                            size=imgsize, align_corners=False)

                result = torch.squeeze(result).cpu().numpy()

            for k, img_name in enumerate(img_names):
                img_name = img_name[0]
                sub_dirs = img_name.split('/')
                clip_dir = os.path.join(img_dir, sub_dirs[folder_index])
                if not os.path.exists(clip_dir):
                    os.makedirs(clip_dir)
                sal_path = os.path.join(clip_dir, sub_dirs[-1].replace('.jpg', '.png'))

                result_k = Image.fromarray((result[k] * 255).astype(np.uint8))
                result_k.save(sal_path)
            if idx % 100 == 0:
                runinfo = "Running test [%d/%d]" % (idx + 1, len(test_loader))
                print(runinfo)

if __name__ == '__main__':
    main()
    print('done')
