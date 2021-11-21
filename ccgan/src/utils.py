
import torch
from PIL import Image
from pathlib import Path

from torchvision.utils import save_image


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# get face with a blank inplace of the surgical mask from the groundtruth filename
def get_masked_face(f):
    masked_dataset_path = "../../data/masked"
    masks_dataset_path = "../../data/mask"

    f = Path(f).stem
    mask = Image.open("%s/%s.jpg" % (masks_dataset_path, f))
    masked = Image.open("%s/%s_surgical.jpg" % (masked_dataset_path, f))

    white = Image.new("L", masked.size, 255)
    mask_applied = Image.composite(white, masked, mask)

    return mask_applied


def save_sample(generator, saved_samples, batches_done):
    # Generate inpainted image
    gen_imgs = generator(saved_samples["masked"], saved_samples["lowres"])
    # Save sample
    sample = torch.cat(
        (saved_samples["masked"].data, gen_imgs.data, saved_samples["imgs"].data), -2)
    save_image(sample, "images/%d.png" % batches_done, nrow=5, normalize=True)
