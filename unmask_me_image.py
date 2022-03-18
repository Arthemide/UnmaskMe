# Principal packages
import argparse

import cv2
import torch

from unmask_me_utils import load_models, predict_face

if __name__ == "__main__":
    # the computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device set to:", device)

    mask_detector_model_path = "model_weights/mask_face_detector.pt"
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

    segmentation_model, generator_model = load_models(
        args,
        mask_detector_model_path,
        mask_segmentation_model_path,
        ccgan_path,
        device,
    )

    image = cv2.imread(args["image"])
    if image is not None:
        predict_face(
            image,
            segmentation_model,
            generator_model,
            mask_detector_model_path,
            args["confidence"],
            args["image"],
        )
        # show the output image
        print("[INFO] Job done, showing image")
        cv2.imshow("Output", image)
        cv2.waitKey(0)
    else:
        print("[INFO] Bad image path")

# do a bit of cleanup
cv2.destroyAllWindows()
