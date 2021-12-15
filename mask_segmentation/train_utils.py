import os
import random
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from MaskTheFace.utils.aux_functions import download_dlib_model
from scipy.stats import norm
from model import UNet


def fast_display(*img2dlist):
    """
    used to display multiple image

    Args:
        *img2dlist(List) a list of image
    """
    plt.figure(figsize=(16, 8))
    nbimg = len(img2dlist)
    cols = min(9, nbimg)
    rows = (nbimg // cols) + 1
    for ii, img2d in enumerate(img2dlist):
        plt.subplot(rows, cols, 1 + ii)
        plt.imshow(img2d)
    plt.show()


def my_collate(batch):
    """
    if a value could not have been created (ex: a face not detected) replace it

    Args:
        batch(torch.utils.data.dataloader)

    Returns:
        (torch.utils.data.dataloader)
    """
    len_batch = len(batch)  # original batch length
    batch = list(filter(lambda x: x is not None, batch))  # filter out all the Nones
    if len_batch > len(
        batch
    ):  # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
        diff = len_batch - len(batch)
        for i in range(diff):
            batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)


def load(
    path="../weigth.pth",
    load=True,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    """
    load our serialized face mask segmentation model from disk for training

    Args:
        device: The device to load the models on.
        path (string): path to the model weigth
        load (Boolean): load or not the  pretrained weigth

    Returns:
        (torch.nn.Module) : the Unet
    """
    model = UNet(3, 1).float()
    if load:
        model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


class AddGaussianNoise(object):
    """
    add Gaussian noise

    Args:
        mean (int): mean
        std (int): std
    """

    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class Args:
    """
    args mimic an arg parser to use MaskeTheFace repo


    """

    def __init__(
        self, mask_types=["surgical"], pattern="", color="#0473e2", color_weight=0.5
    ):
        self.mask_types = mask_types
        self.mask_type = mask_types[0]
        self.pattern = pattern
        self.color = color
        self.color_weight = color_weight

    verbose = False
    code = None
    path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(path_to_dlib_model):
        download_dlib_model()
    predictor = dlib.shape_predictor(path_to_dlib_model)
    detector = dlib.get_frontal_face_detector()


# src: https://medium.com/@fanzongshaoxing/adjust-local-brightness-for-image-augmentation-8111c001059b
def _decayed_value_in_norm(x, max_value, min_value, center, range):
    """
    decay from max value to min value following Gaussian/Normal distribution
    """
    radius = range / 3
    center_prob = norm.pdf(center, center, radius)
    x_prob = norm.pdf(x, center, radius)
    x_value = (x_prob / center_prob) * (max_value - min_value) + min_value
    return x_value


def _decayed_value_in_linear(x, max_value, padding_center, decay_rate):
    """
    decay from max value to min value with static linear decay rate.
    """
    x_value = max_value - abs(padding_center - x) * decay_rate
    if x_value < 0:
        x_value = 1
    return x_value


def generate_parallel_light_mask(
    mask_size,
    position=None,
    direction=None,
    max_brightness=255,
    min_brightness=0,
    mode="gaussian",
    linear_decay_rate=None,
):
    """
    Generate decayed light mask generated by light strip given its position, direction

    Args:
        mask_size: tuple of integers (w, h) defining generated mask size
        position: tuple of integers (x, y) defining the center of light strip position,
                  which is the reference point during rotating
        direction: integer from 0 to 360 to indicate the rotation degree of light strip
        max_brightness: integer that max brightness in the mask
        min_brightness: integer that min brightness in the mask
        mode: the way that brightness decay from max to min: linear or gaussian
        linear_decay_rate: only valid in linear_static mode. Suggested value is within [0.2, 2]

    Return:
        light_mask: ndarray in float type consisting value from 0 to strength
    """
    if position is None:
        pos_x = random.randint(0, mask_size[0])
        pos_y = random.randint(0, mask_size[1])
    else:
        pos_x = position[0]
        pos_y = position[1]
    if direction is None:
        direction = random.randint(0, 360)
    if linear_decay_rate is None:
        if mode == "linear_static":
            linear_decay_rate = random.uniform(0.2, 2)
        if mode == "linear_dynamic":
            linear_decay_rate = (max_brightness - min_brightness) / max(mask_size)
    assert mode in [
        "linear_dynamic",
        "linear_static",
        "gaussian",
    ], "mode must be linear_dynamic, linear_static or gaussian"
    padding = int(max(mask_size) * np.sqrt(2))
    # add padding to satisfy cropping after rotating
    canvas_x = padding * 2 + mask_size[0]
    canvas_y = padding * 2 + mask_size[1]
    mask = np.zeros(shape=(canvas_y, canvas_x), dtype=np.float32)
    # initial mask's up left corner and bottom right corner coordinate
    init_mask_ul = (int(padding), int(padding))
    init_mask_br = (int(padding + mask_size[0]), int(padding + mask_size[1]))
    init_light_pos = (padding + pos_x, padding + pos_y)
    # fill in mask row by row with value decayed from center
    for i in range(canvas_y):
        if mode == "linear":
            i_value = _decayed_value_in_linear(
                i, max_brightness, init_light_pos[1], linear_decay_rate
            )
        elif mode == "gaussian":
            i_value = _decayed_value_in_norm(
                i, max_brightness, min_brightness, init_light_pos[1], mask_size[1]
            )
        else:
            i_value = 0
        mask[i] = i_value
    # rotate mask
    rotate_M = cv2.getRotationMatrix2D(init_light_pos, direction, 1)
    mask = cv2.warpAffine(mask, rotate_M, (canvas_x, canvas_y))
    # crop
    mask = mask[init_mask_ul[1] : init_mask_br[1], init_mask_ul[0] : init_mask_br[0]]
    mask = np.asarray(mask, dtype=np.uint8)
    # add median blur
    mask = cv2.medianBlur(mask, 9)
    mask = 255 - mask
    return mask


def add_parallel_light(
    image,
    light_position=None,
    direction=None,
    max_brightness=255,
    min_brightness=0,
    mode="gaussian",
    linear_decay_rate=None,
    transparency=None,
):
    """
    Add mask generated from parallel light to given image
    """
    if transparency is None:
        transparency = random.uniform(0.5, 0.85)
    frame = image
    height, width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = generate_parallel_light_mask(
        mask_size=(width, height),
        position=light_position,
        direction=direction,
        max_brightness=max_brightness,
        min_brightness=min_brightness,
        mode=mode,
        linear_decay_rate=linear_decay_rate,
    )
    hsv[:, :, 2] = hsv[:, :, 2] * transparency + mask * (1 - transparency)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    frame[frame > 255] = 255
    frame = np.asarray(frame, dtype=np.uint8)
    return frame
