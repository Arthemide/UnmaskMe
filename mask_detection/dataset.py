# -*- coding: utf-8 -*-
# Principal packages
import os
import sys
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

current_dir = os.path.dirname(os.path.realpath(__file__))
parrent_dir = os.path.dirname(current_dir)
sys.path.append(parrent_dir)

def get_transforms():
    """
    get transfroms for dataloader

    Args:
        img_size: size of image to train

    Returns:
        transform_x: transform for image
        transform_lr: transform for image in low resolution
    """
    train_transform = transforms.Compose([
        # transforms.CenterCrop(200),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        # transforms.RandomRotation(degrees=(30, 70)),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.5, 0.5, 0.5],
        #     std=[0.5, 0.5, 0.5]
        # )
    ])

    # the validation transforms
    valid_transform = transforms.Compose([
        # transforms.CenterCrop(300),
        # transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.5, 0.5, 0.5],
        #     std=[0.5, 0.5, 0.5]
        # )
    ])
    return train_transform, valid_transform

def get_data_loader(data_path, transform, batch_size, shuffle, num_workers):
    return DataLoader(
        datasets.ImageFolder(
            root=data_path,
            transform=transform
            ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
) 
