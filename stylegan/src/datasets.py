import glob

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self, root, get_mask, transforms_x=None, transforms_lr=None, mode="train"
    ):
        self.transform_x = transforms.Compose(transforms_x)
        self.transform_lr = transforms.Compose(transforms_lr)

        self.files = sorted(glob.glob("%s/*.*" % root))
        self.get_mask = get_mask

    def __getitem__(self, index):

        filename = self.files[index % len(self.files)]

        img = Image.open(filename)

        mask = self.get_mask(filename)

        x = self.transform_x(img)
        x_lr = self.transform_lr(img)

        m_x = self.transform_x(mask)

        return {"x": x, "x_lr": x_lr, "m_x": m_x}

    def __len__(self):
        return len(self.files)


class MaskDataset(Dataset):
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


# Dataset composed of only one image
class UniqueDataset(Dataset):
    def __init__(self, image, transforms_x=None, transforms_lr=None, mode="eval"):
        self.transform_x = transforms.Compose(transforms_x)
        self.transform_lr = transforms.Compose(transforms_lr)

        self.image = image

    def __getitem__(self, index):
        x = self.transform_x(self.image)
        x_lr = self.transform_lr(self.image)

        return {"x": x, "x_lr": x_lr}

    def __len__(self):
        return 1
