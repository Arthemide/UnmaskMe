import os
import dload


def get_celeba():
    if os.path.exists("dataset/celeba"):
        print("Dataset already exists")
        return
    os.makedirs("dataset/", exist_ok=True)
    print("Downloading CelebA dataset...")
    url = "https://link.eu1.storjshare.io/s/jurm4owtgpgrekgmrsvtz67n3wuq/datasets/celeba.zip?wrap=0"
    return dload.save_unzip(url, "dataset/", True)


def get_MaskTheFace():
    if os.path.exists("dataset/MaskTheFace"):
        print("MaskTheFace already exists")
        return
    os.makedirs("dataset/", exist_ok=True)
    print("Cloning MaskTheFace...")
    url = "https://github.com/aqeelanwar/MaskTheFace.git"
    return dload.git_clone(url, "dataset/MaskTheFace")


def get_mask_detector_model():
    if os.path.exists("model_weights/mask_detector_model.pth"):
        print("Mask detector model already exists")
        return
    os.makedirs("model_weights/", exist_ok=True)
    print("Downloading  Mask detector model...")
    url = "https://link.eu1.storjshare.io/juktaddoxro75bg4irc55ewerevq/datasets/model_mask_detector.pth"
    return dload.save(url, "model_weights/mask_detector_model.pth")


def replace_face(image, gan_preds, locations):
    for (box, pred) in zip(locations, gan_preds):
        (startX, startY, endX, endY) = box
        image[startY:endY, startX:endX] = pred
    return image
