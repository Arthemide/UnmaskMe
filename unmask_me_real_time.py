# Principal packages
import argparse
import time

import cv2
import imutils
import torch
from imutils.video import VideoStream

from utils import load_models, predict_face

try:
    get_face_detector_model()
    get_mask_detector_model()
    get_mask_segmentation_model()
    get_ccgan_model()
    get_YOLOv5_repo()
    get_YOLOv5_model()
except:

    print("error")
    raise ValueError("Error while loading models")


from mask_detection.YOLOv5.utils.detect import run_model

if __name__ == "__main__":
    # the computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device set to:", device)

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
        default=0.75,
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

    maskModel, faceNet, segmentation_model, generator_model = load_models(
        args,
        face_detector_path,
        mask_detector_model_path,
        mask_segmentation_model_path,
        ccgan_path,
        device,
    )

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

        predict_face(
            frame,
            faceNet,
            maskModel,
            segmentation_model,
            generator_model,
            args["confidence"],
        )

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
