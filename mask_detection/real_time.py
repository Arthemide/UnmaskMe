# -*- coding: utf-8 -*-
# Principal packages
import torch
from torchvision import transforms
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import model
import utils

output_path = 'mask_detector/'

if __name__ == "__main__":
# the computation device
	device = ('cuda' if torch.cuda.is_available() else 'cpu')
	print("[INFO] Device set to: ", device)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

maskModel, faceNet = utils.load_models(device, args["face"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(faces, locs, preds) = utils.detect_and_predict_mask(frame, faceNet, maskModel, args["confidence"])
	
	# Adrien function inputs: faces = all faces with a mask
	
	utils.display_result(locs, preds, frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
