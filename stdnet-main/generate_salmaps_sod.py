"""
Generating saliency maps of SOD models

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
from utils import state_from_training

from dataloader_sod import prepare_test_data as get_saldata

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='Generate sal maps by SDNet')

parser.add_argument('--savedir', type=str, default='results/savedir', 
        help='path to save result and checkpoint')
parser.add_argument('--datadir', type=str, default='../data', 
        help='dir to the dataset')
parser.add_argument('--testdata', type=str, default='ECSSD', 
        help='name of validation dataset')
parser.add_argument('--evaluate', type=str, default=None, 
        help="full path to checkpoint to be evaluated or 'best'")

parser.add_argument('--model', type=str, default='sdnet', 
        help='model to train the dataset')
parser.add_argument('--bn', action='store_true',
        help='whether use bn in backbone')
parser.add_argument('--train-config', type=str, default='sdnet', 
        help='model configurations during training, please refer to models/config.py for possible configurations')
parser.add_argument('--inference-config', type=str, default='baseline', 
        help='model configurations during inference, please refer to models/config.py for possible configurations')
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
    args.use_cuda = torch.cuda.is_available()

    if args.size is not None:
        args.size = (args.size, args.size)

    print(args)

    ### Create model
    args.config = args.train_config
    model_training_time = getattr(models, args.model)(args)

    args.config = args.inference_config
    args.bn = False
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

    save_dir = args.savedir

    ### Load Data
    testdatas = args.testdata.split('+')
    for tdat in testdatas:
        args.testdata = tdat
        test_loader = get_saldata(args)

        img_dir = os.path.join(save_dir, '{}'.format(args.testdata))
        eval_info = '\nBegin to generating saliency maps for model {}...\nImg generated in {}\n'.format(args.model, img_dir)
        print(eval_info)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        else:
            print('%s already exits, but it will be regenerated' % img_dir)

        for idx, (image, img_name, imgsize) in enumerate(test_loader):

            img_name = img_name[0]
            imgsize = [x.item() for x in imgsize]
            with torch.no_grad():
                image = image.cuda() if args.use_cuda else image
                result = model(image)
                result = F.interpolate(result, mode='bilinear', 
                            size=imgsize, align_corners=False)

                result = torch.squeeze(result).cpu().numpy()

            result = Image.fromarray((result * 255).astype(np.uint8))
            sal_path = os.path.join(img_dir, "%s.png" % img_name)
            gt_path = os.path.join(args.datadir, args.testdata, 'GT', "%s.png" % img_name)
            result.save(sal_path)
            if idx % 100 == 0:
                runinfo = "Running test [%d/%d]" % (idx + 1, len(test_loader))
                print(runinfo)

if __name__ == '__main__':
    os.makedirs(args.savedir, exist_ok=True)
    main()
    print('done')
