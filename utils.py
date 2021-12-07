import os
import dload


def get_celeba():
    path = "dataset/celeba"
    if os.path.exists(path):
        return path
    os.makedirs(path.split("/")[0], exist_ok=True)
    print("Downloading CelebA dataset...")
    url = "https://link.eu1.storjshare.io/s/jurm4owtgpgrekgmrsvtz67n3wuq/datasets/celeba.zip?wrap=0"
    return dload.save_unzip(url, path.split("/")[0], True)


def get_dataset():
    path = "dataset/dataset"
    if os.path.exists(path):
        return path
    os.makedirs(path.split("/")[0], exist_ok=True)
    print("Downloading dataset...")
    url = "https://link.eu1.storjshare.io/jxjaaumkj2zlbsadwkbu2dr4p7dq/datasets/dataset.zip?wrap=0"
    return dload.save_unzip(url, path.split("/")[0], True)


def get_MaskTheFace():
    path = "MaskTheFace"
    if os.path.exists(path):
        return path
    os.makedirs(path.split("/")[0], exist_ok=True)
    print("Cloning MaskTheFace...")
    url = "https://github.com/aqeelanwar/MaskTheFace.git"
    return dload.git_clone(url, path)


def get_mask_detector_model():
    path = "model_weights/mask_detector_model.pth"
    if os.path.exists(path):
        return path
    os.makedirs(path.split("/")[0], exist_ok=True)
    print("Downloading mask detector model...")
    url = "https://link.eu1.storjshare.io/juktaddoxro75bg4irc55ewerevq/datasets/model_mask_detector.pth?wrap=0"
    return dload.save(url, path)


def get_mask_segmentation_model():
    path = "model_weights/model_mask_segmentation.pth"
    if os.path.exists(path):
        return path
    os.makedirs(path.split("/")[0], exist_ok=True)
    print("Downloading mask segmentation model...")
    url = "https://link.eu1.storjshare.io/jxab23e5luqjapxi72yweedmoumq/datasets/model_mask_segmentation.pth?wrap=0"
    return dload.save(url, path)


def get_ccgan_model():
    path = "model_weights/ccgan-110.pth"
    if os.path.exists(path):
        return path
    os.makedirs(path.split("/")[0], exist_ok=True)
    print("Downloading ccgan-110 model...")
    url = "https://link.eu1.storjshare.io/juznbc7nwnpecayfjhu4zmlwhpaa/datasets/ccgan-110.pth?wrap=0"
    return dload.save(url, path)


def replace_face(image, gan_preds, locations):
    for (box, pred) in zip(locations, gan_preds):
        (startX, startY, endX, endY) = box
        image[startY:endY, startX:endX] = pred
    return image
