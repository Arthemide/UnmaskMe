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
    get_YOLOv5_repo,
    get_YOLOv5_model
)

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

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.75,
        help="minimum probability to filter weak detections",
    )

    args = vars(ap.parse_args())

    maskModel, faceNet = mask_utils.load_models(
        device, "model_weights/face_detector", "model_weights/mask_detector_model.pth"
    )
    segmentation_model = segmentation_utils.load_model(
        device, "model_weights/model_mask_segmentation.pth"
    )
    generator_model = gan_utils.load_model("model_weights/ccgan-110.pth", device)
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

        (faces, locs) = run_model(
            weights="./model_weights/mask_face_detector.pt",
            data="./mask_detection/YOLOv5/data/mask_data.yaml",
            conf_thres=args["confidence"],
            img0=frame)
    
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
