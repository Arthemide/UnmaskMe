# -*- coding: utf-8 -*-
# Principal packages
import os

import cv2
import numpy as np
import torch
from torchvision import transforms

from mask_detection import model as mask_model


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


def display_result(locations, predictions, frame):
    """
    Display the results.

    Args:
        locations (list): a list of bounding boxes
        predictions (list): a list of predictions
        frame (numpy.ndarray): an image of shape (H, W, 3)

    Returns:
        numpy.ndarray: an image of shape (H, W, 3)
    """
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
    return (faces, locs, preds)
