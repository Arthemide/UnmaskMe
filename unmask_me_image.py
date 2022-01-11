# Principal packages
import argparse

import cv2
import torch

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

    try:
        if face_detector_path == args["face_detector_path"]:
            get_face_detector_model()
        if mask_detector_model_path == args["mask_detector_model_path"]:
            get_mask_detector_model()
        if mask_segmentation_model_path == args["mask_segmentation_model_path"]:
            get_mask_segmentation_model()
        if ccgan_path == args["ccgan_path"]:
            get_ccgan_model()
    except:
        print("error")
        raise ValueError("Error while loading models")

    maskModel, faceNet = mask_utils.load_models(
        device, args["face_detector_path"], args["mask_detector_model_path"]
    )
    segmentation_model = segmentation_utils.load_model(
        device, args["mask_segmentation_model_path"]
    )
    generator_model = gan_utils.load_model(args["ccgan_path"], device)
    print("[INFO] Models loaded")

    image = cv2.imread(args["image"])
    if image is not None:
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
    else:
        print("[INFO] Bad image path")

# do a bit of cleanup
cv2.destroyAllWindows()
