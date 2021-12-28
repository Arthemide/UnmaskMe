# Principal packages
import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim

# Helper libraries
from utils import save_model, save_plots, EarlyStopping, LRScheduler, train, validate
from dataset import get_transforms, get_data_loader
from model import FaceMaskDetectorModel
from dataset import utils
from ressources import get_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=2,
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
        default="../dataset/dataset",
        help="Path to original dataset",
    )
    parser.add_argument(
        "--masks_path",
        type=str,
        default="../dataset/masks_samples",
        help="Path to dataset of masks",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../model_weights/ccgan-110.pth",
        help="interval between image sampling",
    )
    parser.add_argument(
        "--sample_path", type=str, default="images", help="Path to save sample images"
    )
    parser.add_argument(
        "--output_path", type=str, default="models", help="Path to save trained model"
    )
    parser.add_argument(
        "--load_data",
        type=bool,
        default=True,
        help="Set to false to use personnal data",
    )
    opt = parser.parse_args()
    n_epochs = opt.n_epochs
    batch_size = opt.batch_size
    lr = opt.lr
    n_cpu = opt.n_cpu
    # img_size = opt.img_size
    # channels = opt.channels
    # sample_interval = opt.sample_interval
    dataset_path = opt.dataset_path
    masks_path = opt.masks_path
    model_path = opt.model_path
    output_path = opt.output_path
    sample_path = opt.sample_path

    # Loading data
    if opt.load_data:
        dataset_path = get_dataset(dataset_path)
        utils.split_dataset(dataset_path, dataset_path)
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
    if sample_path is not None:
        os.makedirs(sample_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FaceMaskDetectorModel().to(device)
    print(model)

    # Total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizers
    print("INFO: Initializing optimizer")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate
    print("INFO: Initializing learning rate scheduler")
    scheduler = LRScheduler(optimizer)

    # Early Stopping
    print("INFO: Initializing early stopping")
    early_stopping = EarlyStopping()

    # Loss function
    print("INFO: Initializing criterion")
    criterion = nn.CrossEntropyLoss()

    train_transform, valid_transform = get_transforms()

    train_loader = get_data_loader(
        dataset_path + "/training", train_transform, batch_size, True, n_cpu
    )
    valid_loader = get_data_loader(
        dataset_path + "/validation", valid_transform, batch_size, False, n_cpu
    )
    test_loader = get_data_loader(
        dataset_path + "/testing", valid_transform, batch_size, False, n_cpu
    )

    # Lists to keep track of losses and accuracies
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # Start the training
    print("[INFO] Start training... Models will be running on %s" % device)
    for epoch in range(n_epochs):
        print(f"[INFO]: Epoch {epoch+1} of {n_epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader, optimizer, criterion, device
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, valid_loader, criterion, device
        )

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(
            f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}"
        )
        print(
            f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}"
        )
        print("-" * 50)

        scheduler(valid_epoch_loss)
        early_stopping(valid_epoch_loss)
        if early_stopping.early_stop:
            break
        time.sleep(5)

    # save the trained model weights
    save_model(n_epochs, model, optimizer, criterion)
    # save the loss and accuracy plots
    save_plots(train_acc, valid_acc, train_loss, valid_loss)
    print("TRAINING COMPLETE")

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Dice loss: {test_loss:.3f}, dice acc: {test_acc:.3f}")
