import os
import random

import cv2
import dlib
import numpy as np
from imutils import face_utils
from MaskTheFace.utils.aux_functions import (
    download_dlib_model,
    get_six_points,
    mask_face,
    rect_to_bb,
    shape_to_landmarks,
)
from torch.utils.data import DataLoader, Dataset


class MaskFace(object):
    """Add a mask to the face in the image"""

    posible_mask_types = ["surgical", "N95", "KN95", "cloth", "gas", "inpaint"]
    path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"

    def __init__(self, mask_list=None):
        """
        Args:

            mask_list (list): List of masks to be used.
        """
        if mask_list is None:
            self.mask_list = self.posible_mask_types
        else:
            for mask in mask_list:
                if mask not in self.posible_mask_types:
                    raise ValueError(
                        f"Mask {mask} is not in the list of possible masks"
                    )
            self.mask_list = mask_list
        if not os.path.exists(self.path_to_dlib_model):
            download_dlib_model()

    def __call__(self, image):
        """
        Args:
            image (cv2 image): Image to be masked.

        Returns:
            cv2 image: the original image
            cv2 image: the mask of the mask
            cv2 image: the masked image
        """
        original_image = image.copy()
        face_detector = dlib.get_frontal_face_detector()
        face_locations = face_detector(image, 1)
        predictor = dlib.shape_predictor(self.path_to_dlib_model)
        mask = np.zeros(image.shape[:2], dtype="uint8")

        for face_location in face_locations:
            shape = predictor(image, face_location)
            shape = face_utils.shape_to_np(shape)
            face_landmarks = shape_to_landmarks(shape)
            face_location = rect_to_bb(face_location)
            six_points_on_face, angle = get_six_points(face_landmarks, image)
            mask_type = self.get_random_mask()
            # print(f"Applying {mask_type} mask")
            image_masked, mask_binary = mask_face(
                image,
                face_location,
                six_points_on_face,
                angle=angle,
                args=None,
                type=mask_type,
            )
            image = image_masked
            mask += mask_binary

        return original_image, np.clip(mask, 0, 1), image

    def get_random_mask(self):
        """
        Returns:
            str: Random mask.
        """
        return random.choice(self.mask_list)


class MaskedFaceDataset(Dataset):
    """Masked face dataset"""

    def __init__(
        self, root, mask_type=None, pre_transform=None, post_transform=None
    ):
        """
        Args:
            root (str): Root directory of the dataset.
            mask_type (str): Mask type.
            pre_transform (callable, optional): Optional transform to be applied on images
                before any other transformation.
            post_transform (callable, optional): Optional transform to be applied on images
                after all transformations.
        """
        self.root = root
        self.image_list = os.listdir(root)
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.maskface = MaskFace(self.mask_type)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.root, self.image_list[idx]))

        if self.pre_transform is not None:
            image = self.pre_transform(image)

        image = self.maskface(image)

        if self.post_transform is not None:
            image = self.post_transform(image)

        return image

    def __repr__(self):
        return f"{self.__class__.__name__}()"


if __name__ == "__main__":
    dataset = MaskedFaceDataset(
        root="dataset/originals",
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for i, (image, mask) in enumerate(dataloader):
        cv2.imshow("image", np.array(image[0]))
        cv2.imshow("mask", np.array(mask[0]))
        cv2.waitKey(0)
        if i == 0:
            break
