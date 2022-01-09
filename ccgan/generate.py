import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ccgan.dataset import MaskDataset
from ccgan.models import Generator
from ccgan.utils import get_transforms
from functools import partial
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode


def load_model(filename, device=None, eval=True):
    """
    Loads a generator from a file.

    Args:
        filename: The path to the file.
        device: The device to load the models on.
        eval: Whether to set the models to eval mode.

    Returns:
        A tuple of the generator
    """
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
    """
    Converts a cv2 image to a PIL image.

    Args:
        img: The cv2 image.

    Returns:
        The PIL image.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def get_color(img):
    l = 80 * img.shape[1] // img.shape[0]
    y = int(img.shape[1] // 2 - l // 2)
    x = int(img.shape[0] // 2 * 0.8 - l // 2)
    img = img[x : x + l, y : y + l]
    average = img.mean(axis=0).mean(axis=0)
    return np.uint8(average)


def get_mask_applied(img, mask):
    """
    Applies a mask to an image.

    Args:
        img: The image.
        mask: The mask.

    Returns:
        The masked image.
    """
    average = get_color(img)
    img = cv2_to_PIL(img)
    mask = mask.resize(img.size)
    white = Image.new("L", img.size, 255)
    average_color = Image.new("RGB", img.size, (average[2], average[1], average[0]))
    res = Image.composite(white, img, mask)
    average_color = Image.composite(average_color, img, mask)
    return (res, average_color)


def get_np_result(image, mask, img, size):
    """
    Converts a tensor to a numpy array.

    Args:
        image: The image (numpy)
        mask: The mask (pillow)
        img: The tensor
        size(tuple of int): The size of the image.

    Returns:
        The numpy array.
    """
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

    res = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)
    mask.close()
    pil_image.close()

    return res


transform_x, transform_lr = get_transforms(128)


def predict(
    generator,
    images,
    masks,
    transforms_x=transform_x,
    transforms_lr=transform_lr,
    apply=get_mask_applied,
):
    """
    Predicts the masks for a set of images.

    Args:
        generator: The generator.
        images: The images.
        masks: The masks.
        transforms_x: The transforms to apply to the images.
        transforms_lr: The transforms to apply to the low resolution images.
        apply: function to apply to get masks applied to images

    Returns:
        The predictions as numpy array inpainted in the original images.
    """
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

    if isinstance(images[0], np.ndarray):
        size = (len(images[0]), len(images[0][0]))
    else:
        size = images[0].size
    print("[INFO] CGan prediction done")

    return list(map(partial(get_np_result, size=size), images, masks, results))
