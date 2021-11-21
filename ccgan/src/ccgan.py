##############################
#   Context-Conditional GAN
##############################

import os
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets import *
from models import *
from utils import *

import torch

# number of epochs of training
n_epochs = 200
# size of the batches
batch_size = 8
# adam: learning rate
lr = 0.0002
# adam: decay of first order momentum of gradient
b1 = 0.5
# adam: decay of first order momentum of gradient
b2 = 0.999
# number of cpu threads to use during batch generation
n_cpu = 8
# size of each image dimension
img_size = 128
# number of image channels
channels = 3
# interval between image sampling
sample_interval = 10000
# set to none or load nth model in load_model_path
load_model_n = 110

dataset_path = "../../data/train"
load_model_path = "../../data/"
save_model_path = "./models/"

os.makedirs("images", exist_ok=True)
os.makedirs(save_model_path, exist_ok=True)

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
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=lr, betas=(b1, b2))


# ---------------------
#  Load models weigth
# ---------------------

begin_epoch = 0

if (load_model_n is not None):
    #filename = "%s/ccgan-%s/ccgan-%s.pth" % (load_model_path, load_model_n, load_model_n)
    filename = "%s/ccgan-%s.pth" % (load_model_path, load_model_n)
    if (os.path.isfile(filename)):
        print("Loading model %s" % filename)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filename, map_location=torch.device(device))
        begin_epoch = checkpoint["epoch"] + 1
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

        generator.train()
        discriminator.train()

    else:
        print("No such file as %s, can't load models" % filename)


# Dataset loader
transforms_ = [
    transforms.Resize((img_size, img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transforms_lr = [
    transforms.Resize((img_size // 4, img_size // 4), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


train_loader = DataLoader(
    ImageDataset(dataset_path, get_mask=get_masked_face,
                 transforms_x=transforms_, transforms_lr=transforms_lr),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

saved_samples = {}
for epoch in range(begin_epoch, n_epochs):
    for i, batch in enumerate(train_loader):
        imgs = batch["x"]
        imgs_lr = batch["x_lr"]
        masked = batch["m_x"]

        # Adversarial ground truths
        valid = Variable(Tensor(
            imgs.shape[0], *discriminator.output_shape).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(
            imgs.shape[0], *discriminator.output_shape).fill_(0.0), requires_grad=False)

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

        if (i % 1000 == 0):
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                  % (epoch, n_epochs, i, len(train_loader), d_loss.item(), g_loss.item()))

        # Save first ten samples
        if not saved_samples:
            saved_samples["imgs"] = real_imgs[:1].clone()
            saved_samples["masked"] = masked[:1].clone()
            saved_samples["lowres"] = imgs_lr[:1].clone()
        elif saved_samples["imgs"].size(0) < 10:
            saved_samples["imgs"] = torch.cat(
                (saved_samples["imgs"], real_imgs[:1]), 0)
            saved_samples["masked"] = torch.cat(
                (saved_samples["masked"], masked[:1]), 0)
            saved_samples["lowres"] = torch.cat(
                (saved_samples["lowres"], imgs_lr[:1]), 0)

        batches_done = epoch * len(train_loader) + i
        if batches_done % sample_interval == 0:
            save_sample(generator, saved_samples, batches_done)
    if save_model_path:
        print("Saving models")
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'd_loss': d_loss,
            'g_loss': g_loss
        }, "%s/ccgan-%s.pth" % (save_model_path, epoch))
