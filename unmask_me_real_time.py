# Principal packages
import argparse
import time

import cv2
import imutils
import torch
from imutils.video import VideoStream

from mask_detection import utils as mask_utils
from mask_segmentation import utils as segmentation_utils
from ccgan import generate as gan_utils
from ressources import (
    replace_face,
    get_face_detector_model,
    get_mask_detector_model,
    get_mask_segmentation_model,
    get_ccgan_model,
)


if __name__ == "__main__":
    # the computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device set to: ", device)

    face_detector_path = "model_weights/face_detector"
    mask_detector_model_path = "model_weights/mask_detector_model.pth"
    mask_segmentation_model_path = "model_weights/model_mask_segmentation.pth"
    ccgan_path = "model_weights/ccgan-110.pth"

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.5,
        help="minimum probability to filter weak detections",
    )
    ap.add_argument(
        "--face_detector_path",
        type=str,
        default=face_detector_path,
        help="Path to face detector model",
    )
    ap.add_argument(
        "--mask_detector_model_path",
        type=str,
        default=mask_detector_model_path,
        help="Path to mask detector model",
    )
    ap.add_argument(
        "--mask_segmentation_model_path",
        type=str,
        default=mask_segmentation_model_path,
        help="Path to mask segmentation model",
    )
    ap.add_argument(
        "--ccgan_path", type=str, default=ccgan_path, help="Path to ccgan model"
    )

    args = vars(ap.parse_args())

    if face_detector_path == args["face_detector_path"]:
        get_face_detector_model()
    if mask_detector_model_path == args["mask_detector_model_path"]:
        get_mask_detector_model()
    if mask_segmentation_model_path == args["mask_segmentation_model_path"]:
        get_mask_segmentation_model()
    if ccgan_path == args["ccgan_path"]:
        get_ccgan_model()

    maskModel, faceNet = mask_utils.load_models(
        device, args["face_detector_path"], args["mask_detector_model_path"]
    )
    segmentation_model = segmentation_utils.load_model(
        device, args["mask_segmentation_model_path"]
    )
    generator_model = gan_utils.load_model(args["ccgan_path"], device)
    print("[INFO] Models loaded")

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
        (faces, locs, preds) = mask_utils.detect_and_predict_mask(
            frame, faceNet, maskModel, args["confidence"]
        )

        if len(faces) != 0:
            # segment the mask on faces
            faces_mask = segmentation_utils.predict(faces, segmentation_model)

            # predict the face underneath the mask
            gan_preds = gan_utils.predict(
                generator=generator_model, images=faces, masks=faces_mask
            )

            image = replace_face(frame, gan_preds, locs)

            # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
