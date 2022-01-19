from mask_segmentation import utils as segmentation_utils
from ccgan import generate as gan_utils
from ressources import (
    replace_face,
    get_mask_segmentation_model,
    get_ccgan_model,
    get_YOLOv5_repo,
    get_YOLOv5_model,
)

try:
    get_YOLOv5_repo()
except:

    print("error")
    raise ValueError("Error while loading models")

from mask_detection.YOLOv5.utils.detect import run_model


def load_models(
    args,
    mask_detector_model_path,
    mask_segmentation_model_path,
    ccgan_path,
    device,
):
    try:
        print(mask_detector_model_path)
        if mask_detector_model_path == args["mask_detector_model_path"]:
            get_YOLOv5_model()
        if mask_segmentation_model_path == args["mask_segmentation_model_path"]:
            get_mask_segmentation_model()
        if ccgan_path == args["ccgan_path"]:
            get_ccgan_model()
    except:
        print("error")
        raise ValueError("Error while loading models")

    segmentation_model = segmentation_utils.load_model(
        device, args["mask_segmentation_model_path"]
    )
    generator_model = gan_utils.load_model(args["ccgan_path"], device)
    print("[INFO] Models loaded")

    return segmentation_model, generator_model


def predict_face(
    image,
    segmentation_model,
    generator_model,
    yolo_model_path,
    confidence,
    image_path,
):
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (faces, locs) = run_model(
        weights=yolo_model_path,
        data="./mask_detection/YOLOv5/data/mask_data.yaml",
        conf_thres=confidence,
        source=image_path,
    )

    if len(faces) != 0:
        # segment the mask on faces
        faces_mask = segmentation_utils.predict(faces, segmentation_model)

        # predict the face underneath the mask
        gan_preds = gan_utils.predict(
            generator=generator_model, images=faces, masks=faces_mask
        )

        image = replace_face(image, gan_preds, locs)
    return image
