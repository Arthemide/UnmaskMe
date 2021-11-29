import torch
from mask_segmentation.model import UNet
from torchvision import transforms

def predict(images, model):
    # define preprocess transforms
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    preds = []
    # apply transform and predict on image separatly
    for image in images:
        t_image = transform(image)
        t_image = torch.unsqueeze(t_image, 0)
        with torch.no_grad():
            pred = model(t_image)
        pred = transforms.ToPILImage(mode="L")(torch.squeeze(pred, 0))
        preds.append(pred)
    return preds


def load_models(device, ModelPath="weigth.pth"):
    # load our serialized face mask segmentation model from disk
    print("[INFO] loading face mask segmentation model...")
    Unet = UNet(3, 1).float()
    Unet.load_state_dict(torch.load(ModelPath, map_location=torch.device(device)))
    Unet.to(device)
    Unet.eval()  # tell the model to not train
    return Unet
