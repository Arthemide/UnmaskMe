##############################
#   Context-Conditional GAN
##############################

import os
import torch
import torchvision.transforms as transforms
from datasets import ImageDataset
# from ccgan.src.models import Discriminator, Generator
from model import Discriminator, Generator
from ilo_stylegan import LatentOptimizer
from utils import get_masked_face, save_sample, weights_init_normal
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

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

config = {
    'model_type': 'stylegan',
    'stylegan': {
        'seed': 42,
        'device': 'cuda',
        'ckpt': 'checkpoint/stylegan2-ffhq-config-f.pt',
        'geocross': 0.01,
        'mse': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'pe': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'dead_zone_linear': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'dead_zone_linear_alpha': 0.1,
        'reference_loss': 0.0,
        'lpips_method': 'fill',
        'cls_name': 'vgg16',
        'fast_compress': False,
        'observed_percentage': 80,
        'image_size': [1024, 1024],
        'mask_black_pixels': True,
        'steps': [30, 30, 30, 30],
        'project': True,
        'lr': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'lr_same_pace': False,
        'start_layer': 0,
        'end_layer': 8,
        'restore': False,
        'saved_noises': ['files/noises.pt', 'files/latent.pt', 'files/gen_outs.pt'],
        'do_project_gen_out': False,
        'do_project_noises': False,
        'do_project_latent': False,
        'max_radius_gen_out': [1000, 1000, 6000],
        'max_radius_noises': [1000, 1000, 6000],
        'max_radius_latent': [100, 1000, 6000],
        'is_video': False,
        'max_frame_radius_gen_out': [200],
        'max_frame_radius_noises': [5],
        'max_frame_radius_latent': [200],
        'video_freq': 30,
        'per_frame_steps': '100',
        'is_sequence': False,
        'input_files': ['files/original/john_cleese_no_nose.png'],
        'output_files': ['out.png'],
        'dataset_type': 'CelebaHQ',
        'is_dataset': False,
        'num_dataset': 1,
        'files_ext': '.png',
        'save_latent': False,
        'save_gif': 1,
        'save_every': 5,
        'save_on_ref': False
    }
}

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
# 1024, 512, 8
generator = Generator(config['image_size'][0], config['image_size'][1], 512, 8)
discriminator = Discriminator(input_shape[1])
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


# Initialize weights
# discriminator.apply(weights_init_normal)

# Optimizers TODO
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

optimizer = LatentOptimizer(config)

# ---------------------
#  Load models weigth
# ---------------------

begin_epoch = 0

if load_model_n is not None:
    # filename = "%s/ccgan-%s/ccgan-%s.pth" % (load_model_path, load_model_n, load_model_n)
    filename = "%s/ccgan-%s.pth" % (load_model_path, load_model_n)
    if os.path.isfile(filename):
        print("Loading model %s" % filename)
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
    ImageDataset(
        dataset_path,
        get_mask=get_masked_face,
        transforms_x=transforms_,
        transforms_lr=transforms_lr,
    ),
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
            save_sample(generator, saved_samples, batches_done)
    if save_model_path:
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
            "%s/ccgan-%s.pth" % (save_model_path, epoch),
        )
