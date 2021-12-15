import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import random
from torch.utils import data
import os
from MaskTheFace.utils.aux_functions import mask_image
import torchvision.transforms.functional as TF
import train_utils

DATAPATH = "../data/img_align_celeba"
NUMWORKERS = 1
BATCHSIZE = 4
NUMEPOCHS = 3


class dataset(data.Dataset):
    def __init__(
        self, src_image, args, train="train"
    ):  # initial logic happens like transform
        self.src_image = src_image
        self.image_paths = os.listdir(src_image)
        self.args = args
        self.train = train
        print(f"number of images for {train}: {self.__len__()}")

    def transform(self, image, mask):
        image = image[:, :, ::-1]
        # add a spotlight on the image.
        image = train_utils.add_parallel_light(image)

        topil = transforms.ToPILImage()
        image, mask = topil(image), topil(mask)

        resize = transforms.Resize(size=(128, 128))
        image, mask = resize(image), resize(mask)

        if random.random() > 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)

        # Transform to tensor
        image, mask = TF.to_tensor(image), TF.to_tensor(mask)
        return image, mask

    def __getitem__(self, index):
        if self.train == "validation":
            index += int(len(self.image_paths) * 0.6)
        elif self.train == "test":
            index += int(len(self.image_paths) * 0.8)
        src = os.path.join(self.src_image, self.image_paths[index])
        self.args.mask_type = random.choice(self.args.mask_types)
        masked, masktype, mask, original = mask_image(src, self.args)
        if len(masked) == 0 or len(mask) == 0:
            return None
        image, mask = self.transform(masked[0], mask[0])
        return image, mask

    def __len__(self):  # return count of sample we have
        if self.train == "train":
            return int(len(self.image_paths) * 0.6)
        return int(len(self.image_paths) * 0.2)


def get_dataloader(batch_size=BATCHSIZE, num_workers=NUMWORKERS):
    """
    fetch the data loader for train/val/test set

    Args:
        batch_size : batch_size
        num_workers : num_workers

    Return:
        trainloader :train dataloader
        valloader : validation dataloader
        testloader : test dataloader
    """
    args = train_utils.Args()
    train_dataset = dataset(DATAPATH, args, "train")
    val_dataset = dataset(DATAPATH, args, "validation")
    test_dataset = dataset(DATAPATH, args, "test")
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_utils.my_collate,
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_utils.my_collate,
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=train_utils.my_collate,
    )
    return trainloader, valloader, testloader


def dice_coef_metric(pred, label):
    intersection = 2.0 * (pred * label).sum()
    union = pred.sum() + label.sum()
    if pred.sum() == 0 and label.sum() == 0:
        return 1.0
    return intersection / union


def dice_coef_loss(pred, label):
    smooth = 1.0
    intersection = 2.0 * (pred * label).sum() + smooth
    union = pred.sum() + label.sum() + smooth
    return 1 - (intersection / union)


def bce_dice_loss(pred, label):
    dice_loss = dice_coef_loss(pred, label)
    bce_loss = nn.BCELoss()(pred, label)
    return dice_loss + bce_loss


def compute_iou(model, loader, device, threshold=0.3):
    valloss = 0
    with torch.no_grad():
        for step, (data, target) in enumerate(tqdm(loader)):
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            loss = dice_coef_metric(out_cut, target.data.cpu().numpy())
            valloss += loss

    return valloss / step


def train_model(
    model, train_loader, val_loader, loss_func, optimizer, scheduler, num_epochs, device
):
    """
    train the model

    Args:
        model : the model to train
        train_loader : the data loader for training value
        val_loader : the data loader for validation value
        loss_func : the loss function
        optimizer : the otpimizer
        scheduler : the scheduler to update the learning rate
        num_epochs : the number of epochs to train
        device : the cpu or gpu to train

    Return:
        loss_history : list of train loss
        train_history : list of Train IOU
        val_history : list of validation IOU
    """
    loss_history = []
    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        model.train()

        losses = []
        train_iou = []

        for i, (image, mask) in enumerate(tqdm(train_loader)):
            image = image.to(device)
            mask = mask.to(device)
            outputs = model(image)
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

            train_dice = dice_coef_metric(out_cut, mask.data.cpu().numpy())
            loss = loss_func(outputs, mask)
            losses.append(loss.item())
            train_iou.append(train_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mean_iou = compute_iou(model, val_loader, device)
        scheduler.step(val_mean_iou)
        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_iou).mean())
        val_history.append(val_mean_iou)

        print("Epoch : {}/{}".format(epoch + 1, num_epochs))
        print(
            "loss: {:.3f} - dice_coef: {:.3f} - val_dice_coef: {:.3f}".format(
                np.array(losses).mean(), np.array(train_iou).mean(), val_mean_iou
            )
        )
    return loss_history, train_history, val_history


def main():
    os.chdir("./MaskTheFace")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = train_utils.load(path="../weigth.pth", load=False, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=3)
    trainloader, valloader, testloader = get_dataloader(
        batch_size=BATCHSIZE, num_workers=NUMWORKERS
    )

    train_loss_list, valid_loss_list, dice_score_list = train_model(
        model,
        trainloader,
        valloader,
        bce_dice_loss,
        optimizer,
        scheduler,
        NUMEPOCHS,
        device,
    )
    torch.save(model.state_dict(), "../weigth.pth")
    metrics = compute_iou(model, testloader, device)
    print(f"[INFO] testing model IOU: {metrics}")


if __name__ == "__main__":
    main()
