"""
Please refer to __init__.py

Author: Zhuo Su
Date: March 16, 2023
"""

import argparse
import os.path
from evaluator_sod import Eval_thread
from dataloader_sod import EvalDataset

parser = argparse.ArgumentParser(description='Evaluation for SOD')

parser.add_argument('--datadir', type=str, default='../data', 
        help='dir to the dataset')
parser.add_argument('--preddir', type=str, default='results/savedir', 
        help='path to save result and checkpoint')
parser.add_argument('--datasets', type=str, default='ECSSD',
        help='data settings for BSDS, Multicue and NYUD datasets')
parser.add_argument('--savedir', type=str, default='results/savedir', 
        help='path to save result and checkpoint')

args = parser.parse_args()

def evaluate(args):

    pred_dir = args.preddir
    gt_dir = args.datadir
    save_dir = args.savedir

    datasets = args.datasets.split('+')

    threads = []
    for dat in datasets:
        if not os.path.exists(os.path.join(pred_dir, dat)):
            continue
        loader = EvalDataset(pred_dir, gt_dir, dat)
        thread = Eval_thread(loader, save_dir, dat, cuda=True)
        threads.append(thread)

    for thread in threads:
        print(thread.run())
        print(thread.loader.c)

if __name__ == '__main__':
    evaluate(args)
