"""
Generating list files for VSOD dataloaders

Author: Zhuo Su
Date: March 16, 2023
"""

import os
import sys
import argparse
import yaml
import skimage
from skimage import io

parser = argparse.ArgumentParser()

parser.add_argument('--root', type=str, default='./data', 
        help='path where you save the datasets')

args = parser.parse_args()
root = args.root

def check_folder(clip_dir, anno_dir, image_ext, label_ext, root):
    assert os.path.exists(os.path.join(root, anno_dir)), 'annotation dir {} does not exist'.format(anno_dir)

    size = set()

    ## check if images exist and the size matches between img and gt
    labels = os.listdir(os.path.join(root, anno_dir))
    labels = [i for i in labels if label_ext in i]
    for lb in labels:
        lb_path = os.path.join(anno_dir, lb)
        im_path = os.path.join(clip_dir, lb.replace(label_ext, image_ext))
        assert os.path.exists(os.path.join(root, im_path)), 'train_video_image: image {} deos not exist\n'.format(im_path)

        img = skimage.img_as_float(io.imread(os.path.join(root, im_path)))
        gt = skimage.img_as_float(io.imread(os.path.join(root, lb_path)))

        im_h, im_w = img.shape[:2]
        gt_h, gt_w = gt.shape[:2]

        assert im_h == gt_h and im_w == gt_w, 'size unmatch'

        size.add(gt.shape)
    assert len(size) == 1, 'multiple sizes in clip {}'.format(clip_dir)
    return size

if __name__ == '__main__':

    config = yaml.full_load(open('dataset/datasets.yaml'))

    lst_train_video = []
    lst_val_video = []
    lst_test_video = []

    size_info = {}

    print('It takes minutes to check folders, please wait for a while. ' 
            'The list files will be written to ./dataset if checked successfully.' 
            'Unique sizes in each dataset will be ouputed.')
    for name, info in config.items():

        image_ext = info['image_ext']
        label_ext = info['label_ext']

        image_dir = info['image_dir']
        label_dir = info['label_dir']

        size = set()

        if 'video_split' not in info.keys():

            images = os.listdir(os.path.join(root, image_dir))
            images = [i for i in images if image_ext in i]

            for im in images:
                im_path = os.path.join(image_dir, im)
                lb_path = os.path.join(label_dir, im.replace(image_ext, label_ext))
                assert os.path.exists(os.path.join(root, lb_path)), 'train_video_image: label path {} does not exit\n'.format(lb_path)
                lst_train_video.append('{} {}\n'.format(im_path, lb_path))
        elif name in ['DAVSOD']:
            for mode, mode_dir in info['video_split'].items():
                folders = os.listdir(os.path.join(root, mode_dir))
                for i, category in enumerate(folders):
                    if i % 10 == 0:
                        print('processing {}/{} clip'.format(i, len(folders)))
                    clip_dir = os.path.join(mode_dir, category, image_dir)
                    anno_dir = os.path.join(mode_dir, category, label_dir)
                    _size = check_folder(clip_dir, anno_dir, image_ext, label_ext, root)
                    getattr(sys.modules[__name__], 'lst_{}_video'.format(mode)).append('{} {} {} {}\n'.format(clip_dir, image_ext, anno_dir, label_ext))
                    size.update(_size)
        else:
            for mode, folders in info['video_split'].items():
                for i, category in enumerate(folders):
                    if i % 10 == 0:
                        print('processing {}/{} clip'.format(i, len(folders)))
                    clip_dir = os.path.join(image_dir, category) if '**' not in image_dir else image_dir.replace('**', category)
                    anno_dir = os.path.join(label_dir, category) if '**' not in label_dir else label_dir.replace('**', category)
                    _size = check_folder(clip_dir, anno_dir, image_ext, label_ext, root)
                    getattr(sys.modules[__name__], 'lst_{}_video'.format(mode)).append('{} {} {} {}\n'.format(clip_dir, image_ext, anno_dir, label_ext))
                    size.update(_size)
        size_info[name] = str(size)
                    
        print(image_dir)

    with open('dataset/train_video_lst.txt', 'w') as f:
        f.writelines(lst_train_video)

    with open('dataset/val_video_lst.txt', 'w') as f:
        f.writelines(lst_val_video)

    with open('dataset/test_video_lst.txt', 'w') as f:
        f.writelines(lst_test_video)

    print(str(size_info))
    print('done')
