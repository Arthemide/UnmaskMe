# -*- coding: utf-8 -*-
# Principal packages
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model class declaration
class FaceMaskDetectorModel(nn.Module):
    def __init__(self):
        super(FaceMaskDetectorModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 5)

        self.fc1 = nn.Linear(256, 50)

        self.pool = nn.MaxPool2d(2, 2)

        self.model = FaceMaskDetectorModel

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.fc1(x)
        return x

    def predict(self, images):
        self.FaceMaskDetectorModel.to(self.device)
        self.FaceMaskDetectorModel.eval()
        preds = self.FaceMaskDetectorModel(images)

        return preds

    """
    Function to save the trained model to disk.
    """

    def save_model(self, epochs, optimizer, criterion, path):
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": criterion,
            },
            path,
        )

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
