import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import pickle
import os
from .spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)

from .NTUARD_Dataset_Train import NTUARD_TRAIN
from .contrastive_dataset_NTU import ContrastiveDataset
import pdb

import torch
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

def get_ucf101(root='Data', frames_path=''):
    from UCF101_Dataset_Train import UCF101TRAIN

    from UCF101_Dataset_Test import UCF101TEST
    ## augmentations
    crop_scales = [1.0]
    for _ in range(1, 5):
        crop_scales.append(crop_scales[-1] * 0.84089641525) ##smallest scale is 0.5

    transform_train = Compose([
            Scale(136),
            MultiScaleRandomCrop(crop_scales, 112),
            RandomHorizontalFlip(),
            ToTensor(1),
        ])
    
    transform_val = transforms.Compose([
            Scale(136),
            CenterCrop(112),
            ToTensor(1),
            transforms.Normalize(mean=normal_mean, std=normal_std)
        ])

    train_dataset = UCF101TRAIN (root=root, train=True, fold=1, transform=transform_train, frames_path=frames_path)
    test_dataset = UCF101TEST(root=root, train=False, fold=1, transform=transform_val, frames_path=frames_path)

    return train_dataset, test_dataset


def get_ntuard(root='Data', frames_path='/datasets/NTU-ARD/frames-240x135', num_clips=1, cross_subject=False, contrastive=False, augment=True, hard_positive=False, random_temporal=True, multiview=False, args=None):
    normal_mean=128.
    normal_std = 128
    ## augmentations
    crop_scales = [1.0]
    for _ in range(1, 5):
        crop_scales.append(crop_scales[-1] * 0.84089641525)  ##smallest scale is 0.5

    transform_train = Compose([
        Scale(136),
        CenterCrop(112),
        ToTensor(1),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])

    transform_val = Compose([
        Scale(136),
        CenterCrop(112),
        ToTensor(1),
        transforms.Normalize(mean=normal_mean, std=normal_std)

    ])
    '''
    transform_contrastive = Compose([
        Scale(136),
        CenterCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], 0.8),
        transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], 0.2),
        transforms.GaussianBlur(112 // 10),
        ToTensor(1),
    ])
    '''
    keys = dir(args)

    train_datasets = []
    transform_contrastive = transform_train
    contrastive_dataset = ContrastiveDataset(root=root, fold=1, transform=transform_contrastive, num_clips=num_clips,
                                             frames_path=frames_path, cross_subject=cross_subject,
                                             hard_positive=hard_positive, random_temporal=random_temporal,
                                             multiview=multiview, args=args)
    train_dataset = NTUARD_TRAIN(root=root, train=True, fold=1, cross_subject=cross_subject, transform=transform_train,
                                 num_clips=num_clips, frames_path=frames_path, args=args if 'pseudo-label' in keys else None)
    test_dataset = NTUARD_TRAIN(root=root, train=False, fold=1, cross_subject=cross_subject, transform=transform_val, num_clips=num_clips, frames_path=frames_path, args= args if 'pseudo-label' in keys else None)
    if contrastive:
        train_datasets.append(contrastive_dataset)
    elif 'combined_multiview_training' in keys and args.combined_multiview_training:
        train_datasets.append(contrastive_dataset)
        train_datasets.append(train_dataset)
    else:
        train_datasets.append(train_dataset)

    if 'semi_supervised_contrastive_joint_training' in keys and args.semi_supervised_contrastive_joint_training:
        train_datasets.append(train_dataset)

    return train_datasets, test_dataset


if __name__ == "__main__":
    print("test")
    import time
    start = time.time()
    try:
        train_dataset, test_dataset=get_ntuard(contrastive=True)
    except Exception as e:
        print(e)
    print("test")
    print("test", train_dataset.__getitem__(0)[0].shape)
    print("after: ", time.time()-start)