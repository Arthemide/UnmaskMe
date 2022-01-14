import torch
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image


def get_transforms(img_size):
    """
    get transfroms for dataloader

    Args:
        img_size: size of image to train

    Returns:
        transform_x: transform for image
        transform_lr: transform for image in low resolution
    """
    transform_x = [
        transforms.Resize((img_size, img_size), InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    transform_lr = [
        transforms.Resize((img_size // 4, img_size // 4), InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transform_x, transform_lr


def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .

    Args:
        m: A module or layer in a model.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# get face with a blank inplace of the surgical mask from the groundtruth filename
def get_masked_face(f, original_path, masks_path):
    """
    get face with a blank inplace of the surgical mask from the groundtruth filename

    Args:
        f: groundtruth filename
        original_path: path to original images
        masks_path: path to mask of original image

    Returns:
        face: face with a blank inplace of the surgical mask
    """

    f = Path(f).stem
    mask = Image.open("%s/%s.jpg" % (masks_path, f))
    original = Image.open("%s/%s.jpg" % (original_path, f))

    white = Image.new("L", original.size, 255)
    mask_applied = Image.composite(white, original, mask)

    return (original, mask_applied)


def save_sample(generator, saved_samples, batches_done, sample_path):
    """
    Save a sample of generated images.

    Args:
        generator: The generator model.
        saved_samples: The directory to save the sample images.
        batches_done: The number of batches done.
        sample_path: path where to save the samples
    """
    if sample_path is not None:
        # Generate inpainted image
        gen_imgs = generator(saved_samples["masked"], saved_samples["lowres"])
        # Save sample
        sample = torch.cat(
            (saved_samples["masked"].data, gen_imgs.data, saved_samples["imgs"].data),
            -2,
        )
        save_image(
            sample, "%s/%d.png" % (sample_path, batches_done), nrow=5, normalize=True
        )
