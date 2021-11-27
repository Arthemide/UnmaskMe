import os

import cv2
import numpy
import torch
import torchvision.transforms as transforms
from ccgan.src.datasets import MaskDataset
from ccgan.src.models import Generator
from functools import partial
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader


def load_models(filename, eval=True, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filename, map_location=torch.device(device))

    generator = Generator((3, 128, 128))

    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator = generator.to(device)

    if eval is True:
        generator.eval()
    else:
        generator.train()

    return generator


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


def cv2_to_PIL(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def get_mask_applied(img, mask):
    img = cv2_to_PIL(img)
    mask = mask.resize(img.size)
    white = Image.new("L", img.size, 255)
    return Image.composite(white, img, mask)


def get_np_result(image, mask, res, size):
    res = transforms.Compose(
        {
            transforms.Resize(size),
            transforms.Normalize((-0.5, -0.5, -0.5), (2, 2, 2)),
        }
    )(res)
    res = transforms.ToPILImage()(torch.squeeze(res, 0))

    mask = mask.resize((size[1], size[0]))

    pil_image = Image.composite(res, cv2_to_PIL(image), mask)
    res = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_BGR2RGB)

    mask.close()
    pil_image.close()

    return res


def predict(
    generator, images, masks, transforms_x=transforms_, transforms_lr=transforms_lr
):
    if len(images) == 0:
        return list()
    generator.eval()

    loader = DataLoader(
        MaskDataset(
            apply=get_mask_applied,
            images=images,
            masks=masks,
            transforms_x=transforms_x,
            transforms_lr=transforms_lr,
        ),
        batch_size=1,
    )

    results = list()
    for i, b in enumerate(loader):
        with torch.no_grad():
            img = generator(Variable(b["x"]), Variable(b["x_lr"]))
            results.append(img)

    size = (len(images[0]), len(images[0][0]))

    return list(map(partial(get_np_result, size=size), images, masks, results))
