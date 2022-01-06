# Principal packages
import argparse

import cv2
import torch

from mask_segmentation import utils as segmentation_utils
from ccgan import generate as gan_utils
from ressources import (
    replace_face,
    get_face_detector_model,
    get_mask_detector_model,
    get_mask_segmentation_model,
    get_ccgan_model,
    get_YOLOv5_repo,
    get_YOLOv5_model,
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
    args = vars(ap.parse_args())

    segmentation_model = segmentation_utils.load_model(
        device, "model_weights/model_mask_segmentation.pth"
    )
    generator_model = gan_utils.load_model("model_weights/ccgan-110.pth", device)
    print("[INFO] Models loaded")

    image = cv2.imread(args["image"])
    if image is not None:
        (faces, locs) = run_model(
            weights="./model_weights/mask_face_detector.pt",
            data="./mask_detection/YOLOv5/data/mask_data.yaml",
            conf_thres=args["confidence"],
            source=args["image"],
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
            print("[INFO] Job done, showing image")
            cv2.imshow("Output", image)
            cv2.waitKey(0)
    else:
        print("[INFO] Bad image path")

# do a bit of cleanup
cv2.destroyAllWindows()
