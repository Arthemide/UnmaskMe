import os
import dload
import os
import random
import time
import pandas as pd
import numpy as np
from shutil import copyfile, rmtree
import tqdm

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
    print('before ', path)
    if os.path.exists(path):
        return path
    print('after ', path)
    print("/".join(path.split("/")[:-1]))
    # os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    os.makedirs(path, exist_ok=True)
    print("Downloading face detector model...")
    url = "https://link.eu1.storjshare.io/s/juv6co67qia72ieiqziwg4ou7lpq/datasets/face_detector.zip?wrap=0"
    # return dload.save_unzip(url, "/".join(path.split("/")[:-1]), True)
    return dload.save_unzip(url, path, True)


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

dest_dir = 'MyDrive/projet_inria/dataset-V2'


def split_dataset(data_dir, dest_dir, training_size=0.6, validation_size=0.2, testing_size=0.2, debug=False):
    """
    Split dataset into training, validation and testing folders.

    Args:
        data_dir (string): Dataset source directory
        dest_dir (string): Dataset destination directory.
        training_size (int): Training percentage size.
        validation_size (int): Validation percentage size.
        testing_size (int): Testing percentage size.

    Returns:
        boolean: Split succeeded or not.
    """
    if(training_size+ validation_size+testing_size != 1):
        print('[ERROR] be careful size must be equal to 100%')
        return 1

    classes=os.listdir(data_dir)
    source_path=[f'{data_dir}/{a}' for a in classes]

    already_created = False

    training_dir=dest_dir+'/training'
    training_dir_paths=[f'{training_dir}/{a}' for a in classes]
    already_created = folder_created(training_dir, training_dir_paths)

    validation_dir=dest_dir+'/validation'
    validation_dir_paths=[f'{validation_dir}/{a}' for a in classes]
    already_created = folder_created(validation_dir, validation_dir_paths)

    testing_dir=dest_dir+'/testing'
    testing_dir_paths=[f'{testing_dir}/{a}' for a in classes]
    already_created = folder_created(testing_dir, testing_dir_paths)
    
    if (not already_created):
        print('[INFO] splitting dataset..')
        for source,train_dir_path,val_dir_path,test_dir_path in zip(source_path,\
                                        training_dir_paths,validation_dir_paths, testing_dir_paths):
                split_data(source,train_dir_path,val_dir_path,test_dir_path, training_size,validation_size,testing_size,debug)
                try:
                    rmtree(source)
                except:
                    raise ValueError('Error deleting directory')
                if (debug):
                    for sub_dir in training_dir_paths:
                        print(sub_dir,':', len(os.listdir(sub_dir)))
                    for sub_dir in validation_dir_paths:
                        print(sub_dir,':', len(os.listdir(sub_dir)))
                    for sub_dir in testing_dir_paths:
                        print(sub_dir,':', len(os.listdir(sub_dir)))

    return 0

def folder_created(data_directory, dir_paths):
    already_created = False 
    if not os.path.exists(data_directory):
        for dir_path in dir_paths:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
    else:
        already_created = True
    return already_created

def split_data(src_path, training_path, validation_path, testing_path, training_size,validation_size,testing_size, debug=False):
    current_class = training_path.split('/')[-1]

    files = []
    for filename in os.listdir(src_path):
        file = src_path +'/'+ filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")
    number_files = len(files)

    training_length = int(number_files* training_size)
    validation_length = int(number_files* validation_size)
    testing_length = int(number_files *testing_size)
    if(debug):
        print( 
        'SOURCE: ',src_path,
        '\nTRAINING: ', training_path, f'({training_length})',
'\nVALIDATION: ',validation_path, f'({validation_length})',
        '\nTESTING: ',testing_path, f'({testing_length})',)
    
    shuffled_set = random.sample(files, number_files)
    training_set = shuffled_set[0:training_length]
    validation_set = shuffled_set[training_length:(training_length+validation_length)]
    testing_set=shuffled_set[:testing_length]

    for filename in tqdm.tqdm(training_set, desc=f"Splitting training set of class {current_class}"):
        this_file = src_path +'/'+ filename
        destination = training_path +'/'+ filename
        copyfile(this_file, destination)

    for filename in tqdm.tqdm(validation_set, desc=f"Splitting validation set of class {current_class}"):
        this_file = src_path +'/'+ filename
        destination = validation_path+'/' + filename
        copyfile(this_file, destination)
        
    for filename in tqdm.tqdm(testing_set, desc=f"Splitting testing set of class {current_class}"):
        this_file = src_path +'/'+ filename
        destination = testing_path+'/' + filename
        copyfile(this_file, destination)
