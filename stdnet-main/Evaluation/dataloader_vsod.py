"""
Please refer to __init__.py

Author: Zhuo Su
Date: March 16, 2023
"""

from torch.utils import data
import os
from PIL import Image

class ClipDataset(data.Dataset):
    def __init__(self, pred_root, label_root):

        pred_names = os.listdir(pred_root)
        pred_names = sorted(pred_names)[1:-1]

        self.image_path = list(
            map(lambda x: os.path.join(pred_root, x), pred_names))
        self.label_path = list(
            map(lambda x: os.path.join(label_root, x), pred_names))
        self.c = 0

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        if pred.size != gt.size:
            self.c += 1
            raise ValueError('size doesn\'t math in {}'.format(self.image_path[item]))
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)

def get_loaderes(pred_dir, gt_dir, dat, datfile):
    pred_root = os.path.join(pred_dir, dat)
    clip_names = os.listdir(pred_root)

    with open(datfile, 'r') as f:
        videos = f.readlines()
        key = '/{}'.format(dat)
        annotations = [l.strip().split()[2] for l in videos if key in l]

    loaderes = []
    for name in clip_names:
        pred_mask_dir = os.path.join(pred_root, name)
        gt_mask_dir = [p for p in annotations if name in p.split('/')]
        loader = ClipDataset(pred_mask_dir, os.path.join(gt_dir, gt_mask_dir[0]))
        loaderes.append((name, loader))

    return loaderes

