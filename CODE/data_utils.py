"""
 > Modules for processing training/input_trainut_test data  
 > Maintainer: https://github.com/xahidbuffon
"""
import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
# from DCP import *
import cv2
# import albumentations as A
import torch
from pathlib import Path
from typing import Optional
 
class UIEBDataset(Dataset):
    """ Common data pipeline to organize and generate
         training pairs for various datasets   
    """
    def __init__(self, root, transform=None, is_test = False):
        
        if transform is not None:
            self.transform = T.Compose(transform)
        else:
            # No legacy augmentations
            # Paper uses flipping and rotation transform to obtain 7 augmented versions of data
            # Rotate by 90, 180, 270 degs, hflip, vflip? Not very clear
            # This is as close as it gets without having to go out of my way to reproduce exactly 7 augmented versions
            self.transform = T.Compose(
                [   T.Resize((256,256)),
                    T.ToTensor(),]
            )

        self.input_files, self.gt_files, self.t_p_files, self.B_p_files = self.get_file_paths(root, is_test)
        self.len = min(len(self.input_files), len(self.gt_files), len(self.t_p_files))

    def __getitem__(self, index):
        
        input_image = Image.open(self.input_files[index % self.len])
        gt_image = Image.open(self.gt_files[index % self.len])
        t_p = Image.open(self.t_p_files[index % self.len])
        B_p = Image.open(self.B_p_files[index % self.len])
        input_image = self.transform(input_image)
        gt_image = self.transform(gt_image)
        t_p = self.transform(t_p)
        B_p = self.transform(B_p)
        return {"inp": input_image, "gt": gt_image, "t": t_p, "B": B_p}

    def __len__(self):
        return self.len

    def get_file_paths(self, root, is_test):
        if is_test == True:
            input_files = sorted(glob.glob(os.path.join(root, 'input') + "/*.*"))
            t_p_files = sorted(glob.glob(os.path.join(root, 't_prior') + "/*.*"))
            B_p_files = sorted(glob.glob(os.path.join(root, 'B_prior') + "/*.*"))
            gt_files = []
        else:
            input_files = sorted(glob.glob(os.path.join(root, 'input') + "/*.*"))
            gt_files = sorted(glob.glob(os.path.join(root, 'gt') + "/*.*"))
            t_p_files = sorted(glob.glob(os.path.join(root, 't_prior') + "/*.*"))
            B_p_files = sorted(glob.glob(os.path.join(root, 'B_prior') + "/*.*"))
        return input_files, gt_files, t_p_files, B_p_files
    

import os.path
import torch
import torch.utils.data as data
from PIL import Image
import random
from random import randrange
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def rotate(img,rotate_index):
    '''
    :return: 8 version of rotating image
    '''
    if rotate_index == 0:
        return img
    if rotate_index==1:
        return img.rotate(90)
    if rotate_index==2:
        return img.rotate(180)
    if rotate_index==3:
        return img.rotate(270)
    if rotate_index==4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==5:
        return img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==6:
        return img.rotate(180).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==7:
        return img.rotate(270).transpose(Image.FLIP_TOP_BOTTOM)


class TrainLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, 'input')
        self.dir_B = os.path.join(self.root, 'gt')
        self.dir_C = os.path.join(self.root, 't_prior')
        self.dir_D = os.path.join(self.root, 'B_prior')

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))
        self.D_paths = sorted(make_dataset(self.dir_D))

        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        A = Image.open(self.A_paths[index]).convert("RGB")
        B = Image.open(self.B_paths[index]).convert("RGB")
        C = Image.open(self.C_paths[index])
        D = Image.open(self.D_paths[index]).convert("RGB")
        # resize
        resized_a = A.resize((280, 280), Image.ANTIALIAS)
        resized_b = B.resize((280, 280), Image.ANTIALIAS)
        resized_c = C.resize((280, 280), Image.ANTIALIAS)
        resized_d = D.resize((280, 280), Image.ANTIALIAS)
        # crop the training image into fineSize
        w, h = resized_a.size
        x, y = randrange(w - self.fineSize + 1), randrange(h - self.fineSize + 1)
        cropped_a = resized_a.crop((x, y, x + self.fineSize, y + self.fineSize))
        cropped_b = resized_b.crop((x, y, x + self.fineSize, y + self.fineSize))
        cropped_c = resized_c.crop((x, y, x + self.fineSize, y + self.fineSize))
        cropped_d = resized_d.crop((x, y, x + self.fineSize, y + self.fineSize))
        # rotate
        rotate_index = randrange(0, 8)
        rotated_a = rotate(cropped_a, rotate_index)
        rotated_b = rotate(cropped_b, rotate_index)
        rotated_c = rotate(cropped_c, rotate_index)
        rotated_d = rotate(cropped_d, rotate_index)
        # transform to (0, 1)
        tensor_a = self.transform(rotated_a)
        tensor_b = self.transform(rotated_b)
        tensor_c = self.transform(rotated_c)
        tensor_d = self.transform(rotated_d)

        # return tensor_a, tensor_b, tensor_c
        return {"inp": tensor_a, "gt": tensor_b, "t": tensor_c, "B": tensor_d}

    def __len__(self):
        return len(self.A_paths)
    
class ValLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, 'input')
        self.dir_B = os.path.join(self.root, 'gt')
        self.dir_C = os.path.join(self.root, 't_prior')
        self.dir_D = os.path.join(self.root, 'B_prior')

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))
        self.D_paths = sorted(make_dataset(self.dir_D))

        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        A = Image.open(self.A_paths[index]).convert("RGB")
        B = Image.open(self.B_paths[index]).convert("RGB")
        C = Image.open(self.C_paths[index])
        D = Image.open(self.D_paths[index]).convert("RGB")
        # resize
        resized_a = A.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
        resized_b = B.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
        resized_c = C.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
        resized_d = D.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
        # transform to (0, 1)
        tensor_a = self.transform(resized_a)
        tensor_b = self.transform(resized_b)
        tensor_c = self.transform(resized_c)
        tensor_d = self.transform(resized_d)

        # return tensor_a, tensor_b, tensor_c
        return {"inp": tensor_a, "gt": tensor_b, "t": tensor_c, "B": tensor_d}

    def __len__(self):
        return len(self.A_paths)