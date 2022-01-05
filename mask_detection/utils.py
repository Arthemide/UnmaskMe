# Principal packages
import os

import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from mask_detection import model as mask_model


# Early stopping
class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


# Learning rate scheduler
class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


# Training
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print("Training")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()

    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100.0 * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation
def validate(model, testloader, criterion, device):
    model.eval()
    print("Validation")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100.0 * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


def predict(image, model):
    """
    Run the image through the model and return the results.

    Args:
        image (numpy.ndarray): an image of shape (H, W, 3)
        model (torch.nn.Module): a PyTorch model

    Returns:
        torch.Tensor: an image of shape (H, W, 3)
    """
    # define preprocess transforms
    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(224), transforms.ToTensor()]
    )

    # convert to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        preds = model(image)
    return preds


def load_models(device, faceModelPath, maskModelPath):
    """
    Load the face detection and mask detection models.

        Args:
            device (torch.device): the device to load the models to
            faceModelPath (str): path to the face detection model
            maskModelPath (str): path to the mask detection model

        Returns:
            torch.nn.Module: a PyTorch model
            torch.nn.Module: a PyTorch model
    """
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([faceModelPath, "deploy.prototxt"])
    weightsPath = os.path.sep.join(
        [faceModelPath, "res10_300x300_ssd_iter_140000.caffemodel"]
    )
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    # initialize the model and load the trained weights
    maskModel = mask_model.FaceMaskDetectorModel().to(device)
    checkpoint = torch.load(maskModelPath, map_location=device)
    maskModel.load_state_dict(checkpoint["model_state_dict"])
    maskModel.eval()

    return maskModel, faceNet


def detect_and_predict_mask(frame, faceNet, maskModel, default_confidence):
    """
    Detect the faces and predict the masks.

    Args:
        frame (numpy.ndarray): an image of shape (H, W, 3)
        faceNet (torch.nn.Module): a PyTorch model
        maskModel (torch.nn.Module): a PyTorch model
        default_confidence (float): the default confidence to use if no mask is detected

    Returns:
        numpy.ndarray: an image of shape (H, W, 3)
        list: a list of bounding boxes
        list: a list of predictions
    """
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] Computing face detections...")
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    # print('default_confidence ' + str(default_confidence))

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # detection
        detection_confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if detection_confidence > default_confidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            if startX > w or endX > w or startY > h or endY > h:
                continue
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]

            # pass the face through the model to determine if the face
            # has a mask or not
            # print('detection_confidence ' + str(detection_confidence))
            # print('face: ' + str((startY,endY, startX,endX)))
            # print('h, w: ' + str((h, w)))
            predictions = predict(face, maskModel)
            _, pred = torch.max(predictions.data, 1)

            # add the face and bounding boxes to their respective
            # lists
            # print('pred: ' + str(pred))
            if pred == 0:
                faces.append(face)
                preds.append(pred)
                locs.append((startX, startY, endX, endY))

    # return a 2-tuple of the face locations and their corresponding
    # locations
    print("[INFO] Find", len(faces), "mask faces")
    return (faces, locs, preds)


def save_model(epochs, model, optimizer, criterion, output_path="models"):
    """
    Function to save the trained model to disk.
    """
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        },
        output_path + "model.pth",
    )


def save_plots(train_acc, valid_acc, train_loss, valid_loss, output_path="models"):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color="green", linestyle="-", label="train accuracy")
    plt.plot(valid_acc, color="blue", linestyle="-", label="validataion accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(output_path + "accuracy.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", linestyle="-", label="train loss")
    plt.plot(valid_loss, color="red", linestyle="-", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path + "loss.png")


def display_result(locations, predictions, frame):
    # loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locations, predictions):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "No Mask" if pred else "Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        # label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(
            frame,
            label,
            (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    return frame
