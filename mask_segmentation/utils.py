from numpy.lib.type_check import imag
import torch
from mask_segmentation.model import UNet
from torchvision import transforms
import cv2
import numpy as np
import torchvision.transforms.functional as TF

def predict(images, model):
    # define preprocess transforms
    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((128, 128)),transforms.ToTensor()]
    )
    # convert to RGB format
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = transform(image)
    preds = []
    for image in images:
        t_image = transform(image)
    # add batch dimension
        t_image = torch.unsqueeze(t_image, 0)
        with torch.no_grad():
            pred = model(t_image)
        preds.append(pred)
    return preds


def load_models(device, ModelPath = 'weigth.pth'):
    # load our serialized face mask segmentation model from disk
    print("[INFO] loading face mask segmentation model...")
    Unet = UNet(3, 1).float()
    Unet.load_state_dict(torch.load(ModelPath))
    Unet.to(device)
    Unet.eval()  # tell the model to not train
    return Unet
