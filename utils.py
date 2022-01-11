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


def load_models(
    args,
    face_detector_path,
    mask_detector_model_path,
    mask_segmentation_model_path,
    ccgan_path,
    device,
):
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

    return maskModel, faceNet, segmentation_model, generator_model


def predict_face(
    image, faceNet, maskModel, segmentation_model, generator_model, confidence
):
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (faces, locs, preds) = mask_utils.detect_and_predict_mask(
        image, faceNet, maskModel, confidence
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
