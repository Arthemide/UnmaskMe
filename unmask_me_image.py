# -*- coding: utf-8 -*-
# Principal packages
import argparse

import cv2
import torch

from mask_detection import utils as mask_utils
from mask_segmentation import utils as segmentation_utils
from ccgan.src import generate as gan_utils
from utils import replace_face

output_path = "mask_detector/"

if __name__ == "__main__":
    # the computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device set to: ", device)

    # read and preprocess the image
    ap = argparse.ArgumentParser()
    # construct the argument parser and parse the arguments
    ap.add_argument("-i", "--image", type=str, help="image path")
    ap.add_argument(
        "-m",
        "--model",
        type=str,
        default="mask_detector.model",
        help="path to trained face mask detector model",
    )
    ap.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.5,
        help="minimum probability to filter weak detections",
    )
    args = vars(ap.parse_args())

    maskModel, faceNet = mask_utils.load_models(device, "mask_detection/face_detector")
    segmentation_model = segmentation_utils.load_models(
        device, "mask_segmentation/weigth.pth"
    )
    generator_model = gan_utils.load_models("ccgan/models/ccgan-110.pth",device)
    print("[INFO] Models loaded")

    image = cv2.imread(args["image"])

    (faces, locs, preds) = mask_utils.detect_and_predict_mask(
        image, faceNet, maskModel, args["confidence"]
    )

    if len(faces) != 0:
        # segment the mask on faces
        faces_mask = segmentation_utils.predict(faces, segmentation_model)

        # predict the face underneath the mask
        gan_preds = gan_utils.predict(
            generator=generator_model, images=faces, masks=faces_mask
        )
        
        image = replace_face(image, gan_preds, locs)

        # show the output image
        cv2.imshow("Output", image)
        cv2.waitKey(0)

# do a bit of cleanup
cv2.destroyAllWindows()
