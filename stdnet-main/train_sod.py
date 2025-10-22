"""
Training script for SOD models

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
import models
from utils import (
    get_model_parm_nums, 
    load_checkpoint, 
    save_checkpoint,
    AverageMeter,
    adjust_learning_rate,
    preload_stage1_weights,
    cross_entropy_loss_sod,
)

from dataloader_sod import prepare_train_data

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='Train SDNet')

parser.add_argument('--savedir', type=str, default='results/savedir', 
        help='path to save result and checkpoint')
parser.add_argument('--datadir', type=str, default='../data', 
        help='dir to the dataset')
parser.add_argument('--traindata', type=str, default='DUTS-TR', 
        help='name of training dataset')
parser.add_argument('--valdata', type=str, default='SOD', 
        help='name of validation dataset, not necessary, as we take the final checkpoint after training')

parser.add_argument('--model', type=str, default='sdnet', 
        help='model to train the dataset')
parser.add_argument('--bn', action='store_true',
        help='use bn in backbone')
parser.add_argument('--config', type=str, default='sdnet', 
        help='model configurations, please refer to models/config.py for possible configurations')
parser.add_argument('--seed', type=int, default=None, 
        help='random seed (default: None)')
parser.add_argument('--gpu', type=str, default='', 
        help='gpus available')

parser.add_argument('--epochs', type=int, default=180, 
        help='number of total epochs to run')
parser.add_argument('--iter-size', type=int, default=1, 
        help='number of samples in each iteration')
parser.add_argument('--batch-size', type=int, default=24, 
        help='number of samples in each batch')
parser.add_argument('--size', type=int, default=320, 
        help='size of samples')
parser.add_argument('--lr', type=float, default=0.001, 
        help='initial learning rate for all weights')
parser.add_argument('--lr-type', type=str, default='multistep', 
        help='learning rate strategy [cosine, multistep]')
parser.add_argument('--lr-steps', type=str, default='90-150', 
        help='steps for multistep learning rate')
parser.add_argument('--wd', type=float, default=1e-4, 
        help='weight decay for all weights')
parser.add_argument('-j', '--workers', type=int, default=4, 
        help='number of data loading workers')
parser.add_argument('--beta', type=float, default=1.1, 
        help='beta on negative annotations')
parser.add_argument('--warmup', type=int, default=5, 
        help='warmup lr')

parser.add_argument('--resume', action='store_true', 
        help='use latest checkpoint if have any')
parser.add_argument('--resume-from', type=str, default=None, 
        help="full path to checkpoint to resume from")
parser.add_argument('--evaluate', type=str, default=None, 
        help="full path to checkpoint to be evaluated or 'best'")
parser.add_argument('--preload', type=str, default=None, 
        help="full path to checkpoint of pretrained model")
parser.add_argument('--print-freq', type=int, default=10, 
        help='print frequency')
parser.add_argument('--save-freq', type=int, default=10, 
        help='save frequency')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

best_mae = 1000000

def main(running_file):

    global args, best_mae

    ### Refine args
    args.sod = True # for salent object detection
    if args.seed is None:
        args.seed = int(time.time())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.use_cuda = torch.cuda.is_available()
    if args.preload is not None:
        args.warmup = 0

    if args.lr_steps is not None and not isinstance(args.lr_steps, list): 
        args.lr_steps = list(map(int, args.lr_steps.split('-'))) 

    if args.size is not None:
        args.size = (args.size, args.size)

    print(args)

    ### Create model
    model = getattr(models, args.model)(args)

    ### Define optimizer
    groupname = ['conv weigths', 'bn weights', 'relu weights', 'scale weights']
    weights = model.get_weights()
    wds = [args.wd, 0.1 * args.wd, 0., 0.5 * args.wd]
    args.lr_factors = [1.0, 1.0, 1.0, 1.0]
    args.scale_id = -1
    param_groups = [{
        'params': weights[i],
        'lr': args.lr,
        'weight_decay': wds[i]
        } for i in range(len(weights))]

    info = ['%s: lr %.6f, wd %.6f' % (groupname[i], param_groups[i]['lr'], param_groups[i]['weight_decay']) 
            for i in range(len(param_groups))]
    info = '\t'.join(info)
    print(info)
    running_file.write('\n%s\n' % info)
    running_file.flush()

    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.99))

    ### Transfer to cuda devices
    if args.use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        print('cuda is used, with %d gpu devices' % torch.cuda.device_count())
    else:
        print('cuda is not used, the running might be slow')

    cudnn.benchmark = True

    ### Load Data
    train_loader, val_loader = prepare_train_data(args)

    ### Create log file
    log_file = os.path.join(args.savedir, '%s_log.txt' % args.model)

    ### Optionally resume from a checkpoint
    args.start_epoch = 0
    if args.resume or (args.resume_from is not None):
        checkpoint = load_checkpoint(args, running_file)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_mae = checkpoint['best_mae']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    ### Preload model before training
    if args.start_epoch == 0 and args.preload is not None:
        preloaded_state = torch.load(args.preload, map_location='cpu')
        preload_stage1_weights(model, preloaded_state)
        info = "successfully preloaded model from {}\n".format(args.preload)
        print(info)
        running_file.write(info)
        running_file.flush()

    ### Train
    saveID = None
    print('current best: %f' % best_mae)

    lossFunc = cross_entropy_loss_sod

    round_factor = 16
    for epoch in range(args.start_epoch, args.epochs):

        # adjust learning rate
        lr_str = adjust_learning_rate(optimizer, epoch, args)

        # train and validate
        scale_series = [1.5, 1.25, 0.75, 1.0]
        for si, scale in enumerate(scale_series): 
            train_loader.dataset.size = [round(scale * s / round_factor) * round_factor for s in args.size]
            loss = train(train_loader, model, lossFunc, optimizer, epoch, running_file, lr_str, args, scale)
        mae = validate(val_loader, model, epoch, running_file, args)

        is_best = mae <= best_mae
        best_mae = min(mae, best_mae)
        log = ("Epoch %02d/%02d: mae %.4f | best MAE %.4f" + \
              " | train loss %.4f | lr %s | Time %s\n") \
              % (epoch, args.epochs, mae, best_mae, loss, \
              lr_str, time.strftime('%Y-%m-%d %H:%M:%S'))
        with open(log_file, 'a') as f:
            f.write(log)

        saveID = save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_mae': best_mae,
            'optimizer': optimizer.state_dict(),
            }, epoch, args.savedir, saveID, 
            is_best, keep_freq=args.save_freq)

    return


def train(train_loader, model, lossFunc, optimizer, epoch, running_file, running_lr, args, scale):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    ## Switch to train mode
    model.train()

    running_file.write('\n%s\n' % str(args))
    running_file.flush()

    wD = len(str(len(train_loader)))
    wE = len(str(args.epochs))

    end = time.time()
    iter_step = 0
    counter = 0
    loss_value = 0
    optimizer.zero_grad()
    for i, data in enumerate(train_loader):

        ## Measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            input = data[0].cuda(non_blocking=True)
            target = data[1].cuda(non_blocking=True)
            mask = data[2].cuda(non_blocking=True) if args.beta is not None else None

        ## Compute output
        output = model(input)
        #_func = F.l1_loss
        loss = lossFunc(output, target, mask)

        counter += 1
        loss_value += loss.item()
        loss = loss / (args.iter_size * args.batch_size)
        loss.backward()
        if counter == args.iter_size:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
            iter_step += 1

            ## Measure accuracy and record loss
            losses.update(loss_value, args.iter_size * args.batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            loss_value = 0

            ## Record
            if iter_step % args.print_freq == 1:
                runinfo = str(('Epoch: [{0:0%dd}/{1:0%dd}][{2:0%dd}/{3:0%dd}]\t' \
                          % (wE, wE, wD, wD) + \
                          ' x{4:.2f} | Time {batch_time.val:.3f}\t' + \
                          'Data {data_time.val:.3f}\t' + \
                          'Loss {loss.val:.4f} (avg:{loss.avg:.4f})\t' + \
                          'lr {lr}\t').format(
                              epoch, args.epochs, iter_step, 
                              len(train_loader)//args.iter_size, scale,
                              batch_time=batch_time, data_time=data_time, 
                              loss=losses, lr=running_lr))
                print(runinfo)
                running_file.write('%s\n' % runinfo)
                running_file.flush()
    if counter > 0:
        print('process last %d samples' % counter)
        optimizer.step()
        optimizer.zero_grad()

    return losses.avg


def validate(val_loader, model, epoch, running_file, args):
    batch_time = AverageMeter()
    mae = AverageMeter()

    ## Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        with torch.no_grad():
            if args.use_cuda:
                target = target.cuda(non_blocking=True)
                input = input.cuda(non_blocking=True)

            ## Compute output
            output = model(input)

        ## Measure MAE
        h, w = target.size()[2:]
        this_output = (F.interpolate(output, mode='bilinear',
                size=(h, w), align_corners=False) * 255.0).int()
        this_output = this_output.float() / 255.0
        this_mae = F.l1_loss(this_output, target, reduction='mean')
        mae.update(this_mae.item(), 1)
            
        ## Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        ## Record
        if i % args.print_freq == 0:
            runinfo = ('Epoch {0:03d} Test: [{1}/{2}]\t' + \
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
                  'MAE {mae.val:.3f} ({mae.avg:.3f})\t').format(
                      epoch, i, len(val_loader), batch_time=batch_time, mae=mae)
            print(runinfo)
            running_file.write('%s\n' % runinfo)
            running_file.flush()

    print(' * MAE {mae.avg:.3f}'.format(mae=mae))

    return mae.avg


if __name__ == '__main__':
    os.makedirs(args.savedir, exist_ok=True)
    running_file = os.path.join(args.savedir, '%s_running-%s.txt' \
            % (args.model, time.strftime('%Y-%m-%d-%H-%M-%S')))
    with open(running_file, 'w') as f:
        main(f)
    print('done')
