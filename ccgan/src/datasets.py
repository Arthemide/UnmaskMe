import glob

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, get_mask, transforms_x=None, transforms_lr=None, mode='train'):
        self.transform_x = transforms.Compose(transforms_x)
        self.transform_lr = transforms.Compose(transforms_lr)

        self.files = sorted(glob.glob('%s/*.*' % root))
        self.get_mask = get_mask

    def __getitem__(self, index):

        filename = self.files[index % len(self.files)]

        img = Image.open(filename)

        mask = self.get_mask(filename)

        x = self.transform_x(img)
        x_lr = self.transform_lr(img)

        m_x = self.transform_x(mask)

        return {'x': x, 'x_lr': x_lr, 'm_x': m_x}

    def __len__(self):
        return len(self.files)


# Dataset composed of only one image
class UniqueDataset(Dataset):
    def __init__(self, image, transforms_x=None, transforms_lr=None, mode='eval'):
        self.transform_x = transforms.Compose(transforms_x)
        self.transform_lr = transforms.Compose(transforms_lr)

        self.image = image

    def __getitem__(self, index):
        x = self.transform_x(self.image)
        x_lr = self.transform_lr(self.image)

        return {'x': x, 'x_lr': x_lr}

    def __len__(self):
        return 1