# Principal packages
import argparse

import cv2
import torch

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

    # read and preprocess the image
    ap = argparse.ArgumentParser()
    # construct the argument parser and parse the arguments
    ap.add_argument("-i", "--image", type=str, help="image path")
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

    image = cv2.imread(args["image"])
    if image is not None:
        predict_face(
            image,
            faceNet,
            maskModel,
            segmentation_model,
            generator_model,
            args["confidence"],
        )
        # show the output image
        print("[INFO] Job done, showing image")
        cv2.imshow("Output", image)
        cv2.waitKey(0)
    else:
        print("[INFO] Bad image path")

# do a bit of cleanup
cv2.destroyAllWindows()
