# Principal packages
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms():
    """
    get transfroms for dataloader

    Args:
        img_size: size of image to train

    Returns:
        transform_x: transform for image
        transform_lr: transform for image in low resolution
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )

    # the validation transforms
    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return train_transform, valid_transform


def get_data_loader(data_path, transform, batch_size, shuffle, num_workers):
    return DataLoader(
        datasets.ImageFolder(root=data_path, transform=transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
