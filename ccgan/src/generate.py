import os
from PIL import Image

import numpy as np

import torch

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.transforms.functional as F
from torchvision.utils import save_image

from models import Generator
from datasets import UniqueDataset


def load_generator(filename, eval=True, device=None):
    if (os.path.isfile(filename)):
        print("Loading model %s" % filename)
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filename, map_location=torch.device(device))

        generator = Generator((3, 128, 128))

        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator = generator.to(device)

        if eval == True:
            generator.eval()
        else:
            generator.train()

        return generator
    else:
        return None


transforms_ = [
    transforms.Resize((128, 128), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transforms_lr = [
    transforms.Resize((128 // 4, 128 // 4), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


def generate_face(generator, mask, masked, transforms_x=transforms_, transforms_lr=transforms_lr):
    generator.eval()
    white = Image.new("L", masked.size, 255)
    mask_applied = Image.composite(white, masked, mask)

    loader = DataLoader(UniqueDataset(image=mask_applied, transforms_x=transforms_, transforms_lr=transforms_lr),
                        batch_size=1)
    for i, b in enumerate(loader):
        with torch.no_grad():
            img = generator(Variable(b['x']), Variable(b['x_lr']))

    pil_image = transforms.Compose({
        transforms.Resize((masked.size[1], masked.size[0])), transforms.Normalize(
            (-0.5, -0.5, -0.5), (2, 2, 2))
    })(img[0])

    pil_image = transforms.ToPILImage()(pil_image)

    res = Image.composite(pil_image, masked, mask)
    return res
