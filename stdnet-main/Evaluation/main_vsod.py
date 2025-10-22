"""
Please refer to __init__.py

Author: Zhuo Su
Date: March 16, 2023
"""

import argparse
import os.path
from evaluator_vsod import Eval_thread
from dataloader_vsod import get_loaderes

parser = argparse.ArgumentParser(description='Evaluation for SOD')

parser.add_argument('--datadir', type=str, default='../data', 
        help='dir to the dataset')
parser.add_argument('--datfile', type=str, default='../dataset/test_video_lst.txt', 
        help='test list file')
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
        logfile = os.path.join(save_dir, 'result_{}.txt'.format(dat))
        loaderes = get_loaderes(pred_dir, gt_dir, dat, args.datfile)
        maeT ,smeasureT, maxfT, fmT = 0., 0., 0., None
        result_info = []
        for i, (name, loader) in enumerate(loaderes):
            print('processing {}/{}: {}'.format(i, len(loaderes), name))
            thread = Eval_thread(loader, save_dir, dat, cuda=True)
            mae, smeasure, maxf, fm = thread.run()
            result_info.append('{}:\tmae {:.4f},\tS-measure {:.4f},\tmax-F {:.4f}\n'.format(name, mae, smeasure, maxf))
            maeT += mae
            smeasureT += smeasure
            maxfT += maxf
            fmT = fm if fmT is None else fmT + fm
        maeT /= len(loaderes)
        smeasureT /= len(loaderes)
        fmT /= len(loaderes)
        maxfT = fmT.max()
        info = 'The whole dataset {}:\tmae {:.4f},\tS-measure {:.4f},\tmax-F {:.4f}\n'.format(dat, maeT, smeasureT, maxfT)
        print(info)
        result_info.append(info)

        with open(logfile, 'w') as f:
            f.writelines(result_info)


if __name__ == '__main__':
    evaluate(args)
