from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image


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


def get_masked_face(f):
    """
    get face with a blank inplace of the surgical mask from the groundtruth filename

    Args:
        f: groundtruth filename

    Returns:
        face: face with a blank inplace of the surgical mask
    """
    masked_dataset_path = "../../data/masked"
    masks_dataset_path = "../../data/mask"

    f = Path(f).stem
    mask = Image.open("%s/%s.jpg" % (masks_dataset_path, f))
    masked = Image.open("%s/%s_surgical.jpg" % (masked_dataset_path, f))

    white = Image.new("L", masked.size, 255)
    mask_applied = Image.composite(white, masked, mask)

    return mask_applied


def save_sample(generator, saved_samples, batches_done):
    """
    Save a sample of generated images.

    Args:
        generator: The generator model.
        saved_samples: The directory to save the sample images.
        batches_done: The number of batches done.
    """
    # Generate inpainted image
    gen_imgs = generator(saved_samples["masked"], saved_samples["lowres"])
    # Save sample
    sample = torch.cat(
        (saved_samples["masked"].data, gen_imgs.data, saved_samples["imgs"].data), -2
    )
    save_image(sample, "images/%d.png" % batches_done, nrow=5, normalize=True)
