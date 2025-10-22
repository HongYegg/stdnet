"""
SOD dataloaders

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
from torch.utils.data import Dataset


def fold_files(foldname):
    """All files in the fold should have the same extern"""
    allfiles = os.listdir(foldname)
    if len(allfiles) < 1:
        return None
    else:
        ext = allfiles[0].split('.')[-1]
        filelist = [
            fname.replace(''.join(['.', ext]), '') for fname in allfiles
        ]
        return ext, filelist


class Augment(object):
    """
    Augment image as well as target(image like array, not box)
    augmentation include Crop Pad and Filp
    """
    def __init__(self, size_h=15, size_w=15, padding=None, p_flip=None):
        super(Augment, self).__init__()
        self.size_h = size_h
        self.size_w = size_w
        self.padding = padding
        self.p_flip = p_flip

    def get_params(self, img):
        im_sz = img.shape[:2]
        row1 = random.randrange(self.size_h)
        row2 = -random.randrange(self.size_h) + im_sz[0]
        col1 = random.randrange(self.size_w)
        col2 = -random.randrange(self.size_w) + im_sz[1]
        if row1 >= row2 or col1 >= col2:
            raise ValueError(
                "Image size too small, please choose smaller crop size")
        padding = None
        if self.padding is not None:
            padding = random.randint(0, self.padding)
        flip_method = None
        if self.p_flip is not None and random.random() < self.p_flip:
            if random.random() < 0.5:
                flip_method = 'lr'
            else:
                flip_method = 'ud'
        return row1, row2, col1, col2, flip_method, padding

    def transform(self,
                  img,
                  row1,
                  row2,
                  col1,
                  col2,
                  flip_method,
                  padding=None):
        """img should be 2 or 3 dimensional numpy array"""
        img = img[row1:row2,
                  col1:col2, :] if len(img.shape) == 3 else img[row1:row2,
                                                                col1:col2]
        if padding is not None:  # TODO: not working yet, fix it later
            pad = transforms.Pad(padding)
            topil = transforms.ToPILImage()
            img = pad(topil(img))
            img = np.array(img)
        if flip_method is not None:
            if flip_method == 'lr':
                img = np.fliplr(img)
            else:
                img = np.flipud(img)
        return img

    def __call__(self, img, target):
        """img and target should have the same spatial size"""
        paras = self.get_params(img)
        img = self.transform(img, *paras)
        target = self.transform(target, *paras)
        return img, target


class SalData(Dataset):
    """Dataset for saliency detection"""
    def __init__(self, dataDir, augmentation=True, mode='train', size=None, beta=None):
        """
        mode can be 'train', 'val', or 'test'
        """
        super(SalData, self).__init__()
        if not os.path.isdir(os.path.join(dataDir, 'images')):
            raise ValueError(
                'Please put your images in folder \'images\' and GT in \'GT\'')
        try:
            ext, self.imgList = fold_files(os.path.join(dataDir, 'images'))
        except FileNotFoundError:
            raise FileNotFoundError('Please put your images in folder \'images\' and GT in \'GT\'')
        self.dataDir = dataDir

        self.augmentation = augmentation
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.size = size
        self.mode = mode
        self.beta = beta
        self.ext = '.%s' % ext

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        imgName = self.imgList[idx]

        img = skimage.img_as_float(
            io.imread(os.path.join(self.dataDir, 'images', imgName + self.ext)))
        imgsize = img.shape
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
            img = np.repeat(img, 3, 2)

        ###### for testing data ############
        if self.mode == 'test':
            shape = imgsize[:2]
            if self.size is not None:
                img = resize(img, (self.size[0], self.size[1]),
                             mode='reflect',
                             anti_aliasing=False)
            img = (img - self.mean) / self.std
            img = np.transpose(img, (2, 0, 1))
            img = img.astype(np.float32)
            return img, imgName, shape
        ####################################

        gt = skimage.img_as_float(
            io.imread(os.path.join(self.dataDir, 'GT', imgName + '.png'), as_gray=True))
        if self.augmentation is True and self.mode == 'train':
            aug = Augment(size_h=15, size_w=15, p_flip=0.5)
            img, gt = aug(img, gt)

        # Normalize image
        if self.size is not None:
            img = resize(img, (self.size[0], self.size[1]),
                         mode='reflect',
                         anti_aliasing=False)
            if self.mode == 'train':
                gt = resize(gt, (self.size[0], self.size[1]),
                            mode='reflect',
                            anti_aliasing=False)
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        gt = gt[np.newaxis, ::]

        img = img.astype(np.float32)
        gt = gt.astype(np.float32)

        if self.mode == 'train' and self.beta is not None:
            num_positive = np.sum(gt>0.8).astype(np.float32)
            num_negative = np.sum(gt<0.01).astype(np.float32)
            pos_weight = num_negative / (num_positive + num_negative)
            neg_weight = self.beta * num_positive / (num_positive + num_negative)
            mask = np.ones_like(gt) * ((pos_weight + neg_weight) / 2)

            #mask[gt > 0.8] = num_negative / (num_positive + num_negative) + 1.0
            mask[gt > 0.8] = pos_weight
            mask[gt < 0.01] = neg_weight
            return img, gt, mask
        else:
            return img, gt


def prepare_train_data(args):

    train_dir = os.path.join(args.datadir, args.traindata)
    val_dir = os.path.join(args.datadir, args.valdata)

    train_set = SalData(train_dir, mode='train', beta=args.beta)
    train_loader = torch.utils.data.DataLoader(
            train_set,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers)

    val_set = SalData(val_dir, mode='val', size=args.size)
    val_loader = torch.utils.data.DataLoader(
            val_set,
            shuffle=False,
            batch_size=1,
            num_workers=args.workers)

    return train_loader, val_loader

def prepare_test_data(args):

    test_dir = os.path.join(args.datadir, args.testdata)

    test_set = SalData(test_dir, mode='test', size=args.size)
    test_loader = torch.utils.data.DataLoader(
            test_set,
            shuffle=False,
            batch_size=1,
            num_workers=args.workers)

    return test_loader
