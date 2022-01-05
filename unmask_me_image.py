# Principal packages
import argparse

import cv2
import torch

from mask_detection import utils as mask_utils
from mask_segmentation import utils as segmentation_utils
from ccgan import generate as gan_utils
from ressources import (
    get_YOLOv5,
    replace_face,
    get_face_detector_model,
    get_mask_detector_model,
    get_mask_segmentation_model,
    get_ccgan_model,
)

try:
    get_face_detector_model()
    get_mask_detector_model()
    get_mask_segmentation_model()
    get_ccgan_model()
    get_YOLOv5()
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
        default=0.5,
        help="minimum probability to filter weak detections",
    )
    ap.add_argument(
        "-d",
        "--data",
        type=str,
        default='./mask_detection/YOLOv5/data/mask_data.yaml',
        help="yaml dataset file path",
    )
    ap.add_argument(
        "-w",
        "--weights",
        type=str,
        default='./model_weights/mask_face_detector.pt',
        help="model weights path",
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

    image = cv2.imread(args["image"])
    if image is not None:
        (faces, locs) = run_model(weights=args["weights"], data=args["data"], conf_thres=args["confidence"], iou_thres=0.45, max_det=1000, source=args["image"])
        frame = mask_utils.display_result(locs, frame=image)
        cv2.imshow("Output", frame)
        cv2.waitKey(0)

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
            # cv2.imshow("Output", image)
            # cv2.waitKey(0)
    else:
        print("[INFO] Bad image path")

# do a bit of cleanup
cv2.destroyAllWindows()
