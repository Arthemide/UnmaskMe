import os
import dload
import os


def get_celeba(path="dataset/celeba"):
    """
    Download and extract the CelebA dataset.

    Returns:
        str: Path to the extracted CelebA dataset.
    """
    if os.path.exists(path):
        return path
    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    print("Downloading CelebA dataset...")
    url = "https://link.eu1.storjshare.io/s/jurm4owtgpgrekgmrsvtz67n3wuq/datasets/celeba.zip?wrap=0"
    return dload.save_unzip(url, "/".join(path.split("/")[:-1]), True)


def get_dataset(path="dataset/dataset"):
    """
    Download and extract the masked dataset.

    Returns:
        str: Path to the extracted masks dataset.
    """
    if os.path.exists(path):
        return path
    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    print("Downloading dataset...")
    url = "https://link.eu1.storjshare.io/jxjaaumkj2zlbsadwkbu2dr4p7dq/datasets/dataset.zip?wrap=0"
    return dload.save_unzip(url, "/".join(path.split("/")[:-1]), True)


def get_masks_samples(path="dataset/masks_samples"):
    """
    Download and extract the celebA masks dataset.

    Returns:
        str: Path to the extracted masked dataset.
    """
    if os.path.exists(path):
        return path
    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    print("Downloading dataset...")
    # url = "https://link.eu1.storjshare.io/jxjaaumkj2zlbsadwkbu2dr4p7dq/datasets/dataset.zip?wrap=0"
    # return dload.save_unzip(url, "/".join(path.split('/')[:-1]), True)


def get_MaskTheFace(path="MaskTheFace"):
    """
    Download and extract the MaskTheFace dataset.

    Returns:
        str: Path to the extracted MaskTheFace dataset.
    """
    if os.path.exists(path):
        return path
    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    print("Cloning MaskTheFace...")
    url = "https://github.com/aqeelanwar/MaskTheFace.git"
    return dload.git_clone(url, path)


def get_face_detector_model(path="model_weights/face_detector"):
    """
    Download and extract the FaceDetector model.

    Returns:
        str: Path to the extracted FaceDetector model.
    """
    if os.path.exists(path):
        return path
    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    print("Downloading face detector model...")
    url = "https://link.eu1.storjshare.io/s/juv6co67qia72ieiqziwg4ou7lpq/datasets/face_detector.zip?wrap=0"
    return dload.save_unzip(url, "/".join(path.split("/")[:-1]), True)


def get_mask_detector_model(path="model_weights/mask_detector_model.pth"):
    """
    Download and extract the MaskDetector model.

    Returns:
        str: Path to the extracted MaskDetector model.
    """
    if os.path.exists(path):
        return path
    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    print("Downloading mask detector model...")
    url = "https://link.eu1.storjshare.io/juktaddoxro75bg4irc55ewerevq/datasets/model_mask_detector.pth?wrap=0"
    return dload.save(url, path)


def get_mask_segmentation_model(path="model_weights/model_mask_segmentation.pth"):
    """
    Download and extract the mask segmentation model.

    Returns:
        str: Path to the extracted mask segmentation model.
    """
    if os.path.exists(path):
        return path
    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    print("Downloading mask segmentation model...")
    url = "https://link.eu1.storjshare.io/jxab23e5luqjapxi72yweedmoumq/datasets/model_mask_segmentation.pth?wrap=0"
    return dload.save(url, path)


def get_ccgan_model(path="model_weights/ccgan-110.pth"):
    """
    Download and extract the ccgan-110 model.

    Returns:
        str: Path to the extracted ccgan-110 model.
    """
    if os.path.exists(path):
        return path
    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    print("Downloading ccgan-110 model...")
    url = "https://link.eu1.storjshare.io/juznbc7nwnpecayfjhu4zmlwhpaa/datasets/ccgan-110.pth?wrap=0"
    return dload.save(url, path)


def replace_face(image, gan_preds, locations):
    """
    Replace the face in the image with the generated predictions.

    Args:
        image (numpy.ndarray): Image to be replaced.
        gan_preds (numpy.ndarray): Predictions from the GAN.
        locations (list): Locations of the face in the image.

    Returns:
        numpy.ndarray: Image with replaced face.
    """
    for (box, pred) in zip(locations, gan_preds):
        (startX, startY, endX, endY) = box
        image[startY:endY, startX:endX] = pred
    return image
