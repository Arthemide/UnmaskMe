
import torch
import model
from torchvision import transforms
import cv2

def predict(image, model):
    # define preprocess transforms
        transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor()
    ])
    # convert to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
    # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            preds = model(image)
        return preds
  
def load_models(device, ModelPath):
    # load our serialized face mask segmentation model from disk
    print("[INFO] loading face mask segmentation model...")
    Unet = model.UNet(3,1).float()
    Unet.load_state_dict(torch.load(ModelPath))
    Unet.to(device)
    Unet.eval() # tell the model to not train
    return Unet
