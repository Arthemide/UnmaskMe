# -*- coding: utf-8 -*-
# Principal packages
import os
import torch
from torchvision import transforms

import model
import utils

# Helper libraries
import argparse
import cv2

import numpy as np

output_path = 'mask_detector/'

if __name__ == "__main__":
    # the computation device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print("[INFO] Device set to: ", device)

    # read and preprocess the image
    ap = argparse.ArgumentParser()
    # construct the argument parser and parse the arguments
    ap.add_argument("-i", "--input", type=str,
	help="image path")
    ap.add_argument("-f", "--face", type=str,
		default="face_detector",
		help="path to face detector model directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
    ap.add_argument("-p", "--production", type=bool, default=False,
		help="is env in production mode")
    args = vars(ap.parse_args())
	
    maskModel, faceNet = utils.load_models(device, args["face"])

    image = cv2.imread(args["input"])

    (_, locs, preds) =  utils.detect_and_predict_mask(image, faceNet, maskModel, args["confidence"])

    utils.display_result(locs, preds, image)
    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

# do a bit of cleanup
cv2.destroyAllWindows()