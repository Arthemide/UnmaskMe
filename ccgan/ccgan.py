##############################
#   Context-Conditional GAN
##############################

import argparse
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torchvision.transforms as transforms
from datasets import ImageDataset
from models import Discriminator, Generator
from pathlib import Path
from PIL import Image
from ressources import get_ccgan_model, get_celeba, get_masks_samples
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
from utils import get_masked_face, save_sample, weights_init_normal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, default=200, help="number of epochs of training"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument(
        "--b1",
        type=float,
        default=0.5,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--img_size", type=int, default=128, help="size of each image dimension"
    )
    parser.add_argument(
        "--channels", type=int, default=3, help="number of image channels"
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1000,
        help="interval between image sampling",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../dataset/celeba",
        help="Path to original dataset",
    )
    parser.add_argument(
        "--masks_path",
        type=str,
        default="../dataset/masks_samples",
        help="Path to dataset of masks",
    )
    parser.add_argument(
        "--model_path", type=str, default="../model_weights/ccgan-110.pth", help="interval between image sampling"
    )
    parser.add_argument(
        "--sample_path", type=str, default="images", help="Path to save sample images"
    )
    parser.add_argument(
        "--output_path", type=str, default="models", help="Path to save trained model"
    )
    parser.add_argument(
        "--load_data", type=bool, default=True, help="Set to false to use personnal data"
    )
    opt = parser.parse_args()
    n_epochs = opt.n_epochs
    batch_size = opt.batch_size
    lr = opt.lr
    b1 = opt.b1
    b2 = opt.b2
    n_cpu = opt.n_cpu
    img_size = opt.img_size
    channels = opt.channels
    sample_interval = opt.sample_interval
    dataset_path = opt.dataset_path
    masks_path = opt.masks_path
    model_path = opt.model_path
    output_path = opt.output_path
    sample_path = opt.sample_path

    # Loading data
    if opt.load_data:
        get_celeba(dataset_path)
        get_masks_samples(masks_path)
        if model_path is not None and Path(model_path).stem == "ccgan-110":
            get_ccgan_model(model_path)
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
    if sample_path is not None:
        os.makedirs(sample_path, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    input_shape = (channels, img_size, img_size)

    # Loss function
    adversarial_loss = torch.nn.MSELoss()


    # Initialize generator and discriminator
    generator = Generator(input_shape)
    discriminator = Discriminator(input_shape)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()


    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # ---------------------
    #  Load models weigth
    # ---------------------

    begin_epoch = 0

    if model_path is not None:
        filename = model_path
        if os.path.isfile(filename):
            print("[LOG] Loading model %s" % filename)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(filename, map_location=torch.device(device))
            begin_epoch = checkpoint["epoch"] + 1
            generator.load_state_dict(checkpoint["generator_state_dict"])
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

            optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
            optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])

            generator.train()
            discriminator.train()

        else:
            print("[ERR] No such file as %s, can't load models" % filename)


    # Dataset loader
    transforms_ = [
        transforms.Resize((img_size, img_size), InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    transforms_lr = [
        transforms.Resize((img_size // 4, img_size // 4), InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]


    train_loader = DataLoader(
        ImageDataset(
            masks_path,
            get_mask=get_masked_face,
            transforms_x=transforms_,
            transforms_lr=transforms_lr,
            original_path=dataset_path,
            masks_path=masks_path
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )


    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    print("[LOG] Start training...")
    print("[LOG] Models will be saved in \"%s\" every epoch and sample generated in \"%s\" every %dth loop" % (output_path, sample_path, sample_interval))
    saved_samples = {}
    for epoch in range(begin_epoch, n_epochs):
        for i, batch in enumerate(train_loader):
            imgs = batch["x"]
            imgs_lr = batch["x_lr"]
            masked = batch["m_x"]

            # Adversarial ground truths
            valid = Variable(
                Tensor(imgs.shape[0], *discriminator.output_shape).fill_(1.0),
                requires_grad=False,
            )
            fake = Variable(
                Tensor(imgs.shape[0], *discriminator.output_shape).fill_(0.0),
                requires_grad=False,
            )

            if cuda:
                imgs = imgs.type(Tensor)
                imgs_lr = imgs_lr.type(Tensor)
                masked = masked.type(Tensor)

            real_imgs = Variable(imgs)
            imgs_lr = Variable(imgs_lr)
            masked = Variable(masked)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(masked, imgs_lr)

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

            if i % 1000 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
                )

            # Save first ten samples
            if not saved_samples:
                saved_samples["imgs"] = real_imgs[:1].clone()
                saved_samples["masked"] = masked[:1].clone()
                saved_samples["lowres"] = imgs_lr[:1].clone()
            elif saved_samples["imgs"].size(0) < 10:
                saved_samples["imgs"] = torch.cat((saved_samples["imgs"], real_imgs[:1]), 0)
                saved_samples["masked"] = torch.cat(
                    (saved_samples["masked"], masked[:1]), 0
                )
                saved_samples["lowres"] = torch.cat(
                    (saved_samples["lowres"], imgs_lr[:1]), 0
                )

            batches_done = epoch * len(train_loader) + i
            if batches_done % sample_interval == 0:
                save_sample(generator, saved_samples, batches_done, sample_path)
        if output_path:
            print("Saving models")
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "d_loss": d_loss,
                    "g_loss": g_loss,
                },
                "%s/ccgan-%s.pth" % (output_path, epoch),
            )
