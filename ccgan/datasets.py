import glob

import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Dataset class for image dataset

    Args:
        root (string): Root directory of dataset
        get_mask (function): Function to get mask from image
        transforms_x (callable, optional): Optional transform to be applied on a sample
        transforms_lr (callable, optional): Optional transform to be applied on a sample
        mode (string, optional): Mode of dataset. Can be "train" or "eval"
    """

    def __init__(
        self,
        root,
        get_mask,
        transforms_x=None,
        transforms_lr=None,
        mode="train",
        original_path=".",
        masks_path=".",
    ):
        self.transform_x = transforms.Compose(transforms_x)
        self.transform_lr = transforms.Compose(transforms_lr)

        self.files = sorted(glob.glob("%s/*.*" % root))
        self.get_mask = get_mask
        self.original_path = original_path
        self.masks_path = masks_path

    def __getitem__(self, index):

        filename = self.files[index % len(self.files)]
        (img, mask) = self.get_mask(filename, self.original_path, self.masks_path)

        x = self.transform_x(img)
        x_lr = self.transform_lr(img)

        m_x = self.transform_x(mask)

        img.close()
        mask.close()

        return {"x": x, "x_lr": x_lr, "m_x": m_x}

    def __len__(self):
        return len(self.files)


class MaskDataset(Dataset):
    """
    Dataset class for mask dataset

        Args:
            images (list): List of images to do prediction onto
            masks (list): List of masks to apply to images, same length than images (sorted like images)
            apply (callable): Function to apply to each image the associated mask
            transforms_x (callable, optional): Optional transform to be applied on a sample
            transforms_lr (callable, optional): Optional transform to be applied on a sample
            mode (string, optional): Mode of dataset. Can be "train" or "eval"
    """

    def __init__(
        self, apply, images, masks, transforms_x=None, transforms_lr=None, mode="train"
    ):
        assert len(images) == len(masks)
        self.transform_x = transforms.Compose(transforms_x)
        self.transform_lr = transforms.Compose(transforms_lr)

        self.images = images
        self.masks = masks

        self.apply = apply

    def __getitem__(self, index):

        mask_applied = self.apply(
            self.images[index % len(self.images)], self.masks[index % len(self.masks)]
        )

        x = self.transform_x(mask_applied)
        x_lr = self.transform_lr(mask_applied)

        return {"x": x, "x_lr": x_lr}

    def __len__(self):
        return len(self.images)
