##############################
#   Context-Conditional GAN
##############################

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets import *
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8,
                    help="size of the batches")
parser.add_argument("--dataset_name", type=str,
                    default="train", help="name of the dataset")
parser.add_argument("--test_dataset_name", type=str,
                    default="test", help="name of the dataset to test")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128,
                    help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=32,
                    help="size of random mask")
parser.add_argument("--channels", type=int, default=3,
                    help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=2000,
                    help="interval between image sampling")
parser.add_argument("--saving_interval", type=int, default=10,
                    help="interval of epoch between model are saved")
parser.add_argument("--save_path", type=str,
                    default="models", help="Path where best models will be saved")
parser.add_argument("--load_models", type=int, default=None,
                    help="To load generator and discriminator number n from save path")
opt = parser.parse_args()
print(opt)


cuda = True if torch.cuda.is_available() else False

input_shape = (opt.channels, opt.img_size, opt.img_size)

# Loss function
adversarial_loss = torch.nn.MSELoss()


# Initialize generator and discriminator
generator = Generator(input_shape)
discriminator = Discriminator(input_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
# Optimizers
optimizer_G = torch.optim.Adam(
generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# ---------------------
#  Load models
# ---------------------

if (opt.load_models is not None):
    if (os.path.isfile("%s/ccgan_generator-%s.pth" % (opt.save_path, opt.load_models))):
        generator = torch.load("%s/ccgan_generator-%s.pth" % (opt.save_path, opt.load_models))
    else:
        print("No such file as %s/ccgan_generator-%s.pth, can't load generator" % (opt.save_path, opt.load_models))
    if (os.path.isfile("%s/ccgan_discriminator-%s.pth" % (opt.save_path, opt.load_models))):
        discriminator = torch.load("%s/ccgan_discriminator-%s.pth" % (opt.save_path, opt.load_models))
    else:
        print("No such file as %s/ccgan_disciminator-%s.pth, can't load discriminator" % (opt.save_path, opt.load_models))

# Dataset loader
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transforms_lr = [
    transforms.Resize((opt.img_size // 4, opt.img_size // 4), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
train_loader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name,
                 transforms_x=transforms_, transforms_lr=transforms_lr),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

test_loader = DataLoader(
    ImageDataset("../../data/%s" % opt.test_dataset_name,
                 transforms_x=transforms_, transforms_lr=transforms_lr, mode='test'),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)



Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def apply_random_mask(imgs):
    idx = np.random.randint(
        0, opt.img_size - opt.mask_size, (imgs.shape[0], 2))

    masked_imgs = imgs.clone()
    for i, (y1, x1) in enumerate(idx):
        y2, x2 = y1 + opt.mask_size, x1 + opt.mask_size
        masked_imgs[i, :, y1:y2, x1:x2] = -1

    return masked_imgs


def save_sample(saved_samples):
    # Generate inpainted image
    gen_imgs = generator(saved_samples["masked"], saved_samples["lowres"])
    # Save sample
    sample = torch.cat(
        (saved_samples["masked"].data, gen_imgs.data, saved_samples["imgs"].data), -2)
    save_image(sample, "images/%d.png" % batches_done, nrow=5, normalize=True)


saved_samples = {}
for epoch in range(opt.n_epochs):
    for i, batch in enumerate(train_loader):
        imgs = batch["x"]
        imgs_lr = batch["x_lr"]

        masked_imgs = apply_random_mask(imgs)

        # Adversarial ground truths
        valid = Variable(Tensor(
            imgs.shape[0], *discriminator.output_shape).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(
            imgs.shape[0], *discriminator.output_shape).fill_(0.0), requires_grad=False)

        if cuda:
            imgs = imgs.type(Tensor)
            imgs_lr = imgs_lr.type(Tensor)
            masked_imgs = masked_imgs.type(Tensor)

        real_imgs = Variable(imgs)
        imgs_lr = Variable(imgs_lr)
        masked_imgs = Variable(masked_imgs)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(masked_imgs, imgs_lr)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
        )


        # Save first ten samples
        if not saved_samples:
            saved_samples["imgs"] = real_imgs[:1].clone()
            saved_samples["masked"] = masked_imgs[:1].clone()
            saved_samples["lowres"] = imgs_lr[:1].clone()
        elif saved_samples["imgs"].size(0) < 10:
            saved_samples["imgs"] = torch.cat(
                (saved_samples["imgs"], real_imgs[:1]), 0)
            saved_samples["masked"] = torch.cat(
                (saved_samples["masked"], masked_imgs[:1]), 0)
            saved_samples["lowres"] = torch.cat(
                (saved_samples["lowres"], imgs_lr[:1]), 0)

        batches_done = epoch * len(train_loader) + i
        if batches_done % opt.sample_interval == 0:
            save_sample(saved_samples)
        if opt.save_path and epoch % opt.saving_interval == 0 and batches_done % opt.sample_interval == 0:
            torch.save(generator, "%s/ccgan_generator-%s.pth" % (opt.save_path, epoch))
            torch.save(discriminator, "%s/ccgan_discriminator-%s.pth" % (opt.save_path, epoch))