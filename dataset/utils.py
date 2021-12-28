import os
import os
import random
from shutil import copyfile, rmtree
import tqdm


def split_dataset(
    data_dir,
    dest_dir,
    training_size=0.6,
    validation_size=0.2,
    testing_size=0.2,
    debug=False,
):
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
    if training_size + validation_size + testing_size != 1:
        print("[ERROR] be careful size must be equal to 100%")
        return 1

    classes = os.listdir(data_dir)
    source_path = [f"{data_dir}/{a}" for a in classes]

    already_created = False

    training_dir = dest_dir + "/training"
    training_dir_paths = [f"{training_dir}/{a}" for a in classes]
    already_created = folder_created(training_dir, training_dir_paths)

    validation_dir = dest_dir + "/validation"
    validation_dir_paths = [f"{validation_dir}/{a}" for a in classes]
    already_created = folder_created(validation_dir, validation_dir_paths)

    testing_dir = dest_dir + "/testing"
    testing_dir_paths = [f"{testing_dir}/{a}" for a in classes]
    already_created = folder_created(testing_dir, testing_dir_paths)

    if not already_created:
        print("[INFO] splitting dataset..")
        for source, train_dir_path, val_dir_path, test_dir_path in zip(
            source_path, training_dir_paths, validation_dir_paths, testing_dir_paths
        ):
            split_data(
                source,
                train_dir_path,
                val_dir_path,
                test_dir_path,
                training_size,
                validation_size,
                testing_size,
                debug,
            )
            try:
                rmtree(source)
            except:
                raise ValueError("Error deleting directory")
            if debug:
                for sub_dir in training_dir_paths:
                    print(sub_dir, ":", len(os.listdir(sub_dir)))
                for sub_dir in validation_dir_paths:
                    print(sub_dir, ":", len(os.listdir(sub_dir)))
                for sub_dir in testing_dir_paths:
                    print(sub_dir, ":", len(os.listdir(sub_dir)))

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


def split_data(
    src_path,
    training_path,
    validation_path,
    testing_path,
    training_size,
    validation_size,
    testing_size,
    debug=False,
):
    current_class = training_path.split("/")[-1]

    files = []
    for filename in os.listdir(src_path):
        file = src_path + "/" + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")
    number_files = len(files)

    training_length = int(number_files * training_size)
    validation_length = int(number_files * validation_size)
    testing_length = int(number_files * testing_size)
    if debug:
        print(
            "SOURCE: ",
            src_path,
            "\nTRAINING: ",
            training_path,
            f"({training_length})",
            "\nVALIDATION: ",
            validation_path,
            f"({validation_length})",
            "\nTESTING: ",
            testing_path,
            f"({testing_length})",
        )

    shuffled_set = random.sample(files, number_files)
    training_set = shuffled_set[0:training_length]
    validation_set = shuffled_set[
        training_length : (training_length + validation_length)
    ]
    testing_set = shuffled_set[:testing_length]

    for filename in tqdm.tqdm(
        training_set, desc=f"Splitting training set of class {current_class}"
    ):
        this_file = src_path + "/" + filename
        destination = training_path + "/" + filename
        copyfile(this_file, destination)

    for filename in tqdm.tqdm(
        validation_set, desc=f"Splitting validation set of class {current_class}"
    ):
        this_file = src_path + "/" + filename
        destination = validation_path + "/" + filename
        copyfile(this_file, destination)

    for filename in tqdm.tqdm(
        testing_set, desc=f"Splitting testing set of class {current_class}"
    ):
        this_file = src_path + "/" + filename
        destination = testing_path + "/" + filename
        copyfile(this_file, destination)
