import cv2
import numpy
import torch
import torchvision.transforms as transforms
from ccgan.datasets import MaskDataset
from ccgan.models import Generator
from ccgan.utils import get_transforms
from functools import partial
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode


def load_model(filename, device=None, eval=True):
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


def cv2_to_PIL(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def get_mask_applied(img, mask):
    img = cv2_to_PIL(img)
    mask = mask.resize(img.size)
    white = Image.new("L", img.size, 255)
    return Image.composite(white, img, mask)


def get_np_result(image, mask, img, size):
    inverseTransform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[2, 2, 2]),
            transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0]),
            transforms.Resize(size, InterpolationMode.BICUBIC),
        ]
    )
    img = inverseTransform(img)
    img = transforms.ToPILImage()(torch.squeeze(img, 0))

    mask = mask.resize((size[1], size[0]))

    pil_image = Image.composite(img, cv2_to_PIL(image), mask)

    res = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_BGR2RGB)

    mask.close()
    pil_image.close()

    return res


transforms_, transforms_lr = get_transforms(128)


def predict(
    generator,
    images,
    masks,
    transforms_x=transforms_,
    transforms_lr=transforms_lr,
    apply=get_mask_applied,
):
    if len(images) == 0:
        return list()
    generator.eval()

    loader = DataLoader(
        MaskDataset(
            apply=apply,
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

    if isinstance(images[0], numpy.ndarray):
        size = (len(images[0]), len(images[0][0]))
    else:
        size = images[0].size

    return list(map(partial(get_np_result, size=size), images, masks, results))
