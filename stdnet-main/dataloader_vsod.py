"""
VSOD dataloaders

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

class VSodVideo(Dataset):
    """
    mode can be 'train', 'val', or 'test'.
    In our implementation:
        train: video frames from training set in (DAVIS2016 + DAVSOD + VOS)
        val: video frames from testing set in DAVIS2016
        test: video frames from testing set in DAVIS2016
        cut: reduce number of imgs for faster validation, 1.0 means no cut
    """
    def __init__(self, root, data_lst, nframes, framegap, augmentation=True, mode='train', size=None, dataset='all', cut=1.1, beta=None):
        super(VSodVideo, self).__init__()

        assert nframes > 1, 'expect nframes larger than 1, got {}'.format(nframes)
        self.nframes = nframes
        self.eval_gap = framegap
        self.cut = cut 
        self.beta = beta

        self.root = root
        self.augmentation = augmentation
        self.mode = mode
        self.size = size
        self.dataset = dataset

        self.mean = [0.485, 0.456, 0.406] * self.nframes
        self.std = [0.229, 0.224, 0.225] * self.nframes

        with open(data_lst, 'r') as f:
            videos = f.readlines()

        if dataset in ['all']:
            self.videos = [l.strip() for l in videos]
        else:
            key = '/{}'.format(dataset)
            self.videos = [l.strip() for l in videos if key in l]

        self.clips = self.get_clips()

    def get_clips(self):
        clips = []
        for line in self.videos:
            line_info = line.split()
            if len(line_info) == 2:
                img, label = line_info
                boring_video_clip = [[os.path.join(self.root, img)],
                        [os.path.join(self.root, label)]]
                clips.append(boring_video_clip)
                continue

            video, img_ext, annotation, label_ext = line_info
            if self.mode in ['test', 'val'] or '/VOS' in video:
                framegaps = [self.eval_gap]
            else:
                framegaps = [1, 4, 8]
            for framegap in framegaps:
                img_dir = os.path.join(self.root, video)
                label_dir = os.path.join(self.root, annotation)
                labels = os.listdir(label_dir)
                labels = [lb for lb in labels if label_ext in lb]
                labels = sorted(labels)
                num_frames = len(labels)
                if self.cut < 1.0:
                    num_frames = int(num_frames * self.cut)
                    labels = labels[:num_frames]
                #assert self.nframes <= num_frames, 'number of annotated frames {} is not enough in {}, at least {}'.format(num_frames, annotation, self.nframes)
                id_start, id_end = [int(lb[:-len(label_ext)].split('_')[-1]) for lb in [labels[0], labels[-1]]]
                gap = (id_end - id_start) / (len(labels) - 1)
                sample_rate = max(1, int(framegap // gap))

                imgs = [os.path.join(img_dir, lb.replace(label_ext, img_ext)) for lb in labels]
                labels = [os.path.join(label_dir, lb) for lb in labels]

                if self.nframes > num_frames:
                    while len(imgs) < self.nframes:
                        imgs.extend(imgs[-2::-1])
                        labels.extend(labels[-2::-1])
                    clips.append([imgs[:self.nframes], labels[:self.nframes]])
                else:
                    sample_rate = min(num_frames // self.nframes, sample_rate)
                    num_chunk = sample_rate * self.nframes
                    chunk_id = 0
                    while (chunk_id + 1) * num_chunk <= num_frames:
                        chunk_start = chunk_id * num_chunk
                        chunk_end = chunk_start + num_chunk
                        chunk_imgs = imgs[chunk_start: chunk_end]
                        chunk_labels = labels[chunk_start: chunk_end]
                        chunk_clips = self.get_chunk_clips(chunk_imgs, chunk_labels)
                        clips.extend(chunk_clips)
                        chunk_id += 1
                    if num_frames - chunk_id * num_chunk > 0:
                        chunk_start = num_frames - num_chunk
                        chunk_end = num_frames
                        chunk_imgs = imgs[chunk_start: chunk_end]
                        chunk_labels = labels[chunk_start: chunk_end]
                        chunk_clips = self.get_chunk_clips(chunk_imgs, chunk_labels)
                        clips.extend(chunk_clips)
        return clips

    def get_chunk_clips(self, imgs, labels):
        assert len(imgs) % self.nframes == 0, 'wrong in processing chunks'

        clips = []
        sample_rate = len(imgs) // self.nframes
        for i in range(sample_rate):
            clips.append([imgs[i::sample_rate], labels[i::sample_rate]])

        return clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        img_paths, gt_paths = self.clips[index]
        clip_folder_name = img_paths[0].split('/')[-2]

        _img = []
        for img_path in img_paths:
            img = skimage.img_as_float(io.imread(img_path))
            imgsize = img.shape
            if img.ndim == 2:
                img = img[:, :, np.newaxis]
                img = np.repeat(img, 3, 2)
            _img.append(img)
        img = np.concatenate(_img, 2)
        if len(img_paths) == 1:
            h, w, _ = img.shape
            img = img.reshape(h, w, 1, 3)
            img = np.repeat(img, self.nframes, 2)
            img = img.reshape(h, w, -1)

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
            return img, img_paths, shape
        ####################################

        _gt = []
        for gt_path in gt_paths:
            gt = skimage.img_as_float(io.imread(gt_path))
            gt = gt[:, :, np.newaxis]
            _gt.append(gt)
        gt = np.concatenate(_gt, 2)
        if len(gt_paths) == 1:
            gt = np.repeat(gt, self.nframes, 2)

        if self.augmentation is True and self.mode == 'train':
            aug = Augment(size_h=15, size_w=15, p_flip=0.5)
            img, gt = aug(img, gt)

        # Normalize image
        if self.size is not None:
            img = resize(img, (self.size[0], self.size[1]),
                         mode='reflect',
                         anti_aliasing=False)
            if self.mode == 'train' or self.dataset in ['VOS']:
                gt = resize(gt, (self.size[0], self.size[1]),
                            mode='reflect',
                            anti_aliasing=False)
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        gt = np.transpose(gt, (2, 0, 1))

        img = img.astype(np.float32)
        gt = gt.astype(np.float32)

        if self.mode == 'train':
            if self.beta is not None:
                if len(self.beta) == 1:
                    beta = float(self.beta[0])
                    num_positive = np.sum(gt>0.8).astype(np.float32)
                    num_negative = np.sum(gt<0.01).astype(np.float32)
                    #pos_weight = num_negative / (num_positive + num_negative)
                    #neg_weight = beta * num_positive / (num_positive + num_negative)
                    pos_weight = num_negative / (beta * num_positive)
                    neg_weight = 1.0
                    mask = np.ones_like(gt) * ((pos_weight + neg_weight) / 2)

                    #mask[gt > 0.8] = num_negative / (num_positive + num_negative) + 1.0
                    mask[gt > 0.8] = pos_weight
                    mask[gt < 0.01] = neg_weight
                else:
                    data_name, beta = self.beta
                    if '/{}/'.format(data_name) in img_paths[0]:
                        mask = np.ones(gt.shape[0]) * float(beta)
                    else:
                        mask = np.ones(gt.shape[0])
                mask = mask.astype(np.float32)
                return img, gt, mask
            else:
                return img, gt
        else:
            #return img, gt, clip_folder_name
            return img, gt


class VSodImage(Dataset):
    """
    Regarding all data as images, not videos. The image can be from a 2D sal dataset, or frames in 3D videos.
    mode can be 'train', 'val', or 'test'.
    In our implementation:
        train: DUTS-TR + video frames from training set in (DAVIS2016 + DAVSOD + VOS)
        val: video frames from testing set in DAVIS2016
        test: video frames from testing set in DAVIS2016
    """
    def __init__(self, root, data_lst, augmentation=True, mode='train', size=None, dataset='all'):
        super(VSodImage, self).__init__()

        self.root = root
        self.augmentation = augmentation
        self.mode = mode
        self.size = size

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        with open(data_lst, 'r') as f:
            files = f.readlines()

        if dataset in ['all']:
            self.files = [l.strip() for l in files]
        else:
            key = '/{}'.format(dataset)
            self.files = [l.strip() for l in files if key in l]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path, gt_path = self.files[index].split()
        img_path = os.path.join(self.root, img_path)
        gt_path = os.path.join(self.root, gt_path)

        img = skimage.img_as_float(io.imread(img_path))
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
            return img, img_path, shape
        ####################################

        gt = skimage.img_as_float(io.imread(gt_path))
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

        return img, gt


def prepare_train_data(args):

    root = args.datadir
    train_lst = args.train_lst
    val_lst = args.val_lst

    if hasattr(args, 'nframes') and args.nframes > 1:
        framegap = args.framegap if hasattr(args, 'framegap') else 1
        train_set = VSodVideo(root, train_lst, args.nframes, framegap, mode='train', beta=args.beta)
        _val_set = []
        for dataset, cut in zip(args.val_datasets, args.val_cuts):
            _val_set.append(VSodVideo(root, val_lst, args.nframes, framegap, mode='val', size=args.size, dataset=dataset, cut=cut))
    else:
        train_set = VSodImage(root, train_lst, mode='train')
        _val_set = [VSodImage(root, val_lst, mode='val', size=args.size)]

    train_loader = torch.utils.data.DataLoader(
            train_set,
            shuffle=True,
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=args.workers)

    val_loader = []
    for val_set in _val_set:
        val_loader.append(torch.utils.data.DataLoader(
                val_set,
                shuffle=False,
                batch_size=1,
                drop_last=True,
                num_workers=args.workers))
    if len(val_loader) == 1:
        val_loader = val_loader[0]

    return train_loader, val_loader

def prepare_test_data(args, dataset='all'):

    root = args.datadir
    test_lst = args.test_lst

    if hasattr(args, 'nframes') and args.nframes > 1:
        framegap = args.framegap if hasattr(args, 'framegap') else 1
        test_set = VSodVideo(root, test_lst, args.nframes, framegap, mode='test', size=args.size, dataset=dataset)
    else:
        test_set = VSodImage(root, test_lst, mode='test', size=args.size, dataset=dataset)
    test_loader = torch.utils.data.DataLoader(
            test_set,
            shuffle=False,
            batch_size=1,
            num_workers=args.workers)

    return test_loader
