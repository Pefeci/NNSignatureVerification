"""
@author: Petr Čírtek
"""

import glob
import itertools
import random
import time

import cv2
import numpy as np
import sklearn.utils
import tensorflow as tf
from scipy import ndimage

from functions import plot_images

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
from PIL import Image

from conf import DATASET_NUM_CLASSES, DATASET_SIGNATURES_PER_PERSON


def create_data(data_dir: str, dataset: str = "cedar", is_genuine: bool = True, gdps_size: int = None):
    is_all = False
    is_all_test = False
    if dataset == "all_test":
        is_all_test = True
    num_classes = DATASET_NUM_CLASSES[dataset]
    persons = []
    index = 0
    if dataset == "all" or dataset == "all_test":
        if gdps_size is None:
            gdps_size = 200
        is_all = True
        if is_all_test:
            dataset = "cedar_test"
        else:
            dataset = "cedar"
        num_classes = DATASET_NUM_CLASSES[dataset]

    if dataset == "cedar" or dataset == "cedar_test":
        if is_genuine:
            images = glob.glob(data_dir + "/cedar/genuine/*.png")
        else:
            images = glob.glob(data_dir + "/cedar/forgery/*.png")
        num_of_signatures = int(len(images) / num_classes)  # this only works with Cedar
        if num_of_signatures == 0:
            num_of_signatures = 1
        for person in range(num_classes):
            signatures = []
            for signature in range(num_of_signatures):
                signatures.append(images[index])
                index += 1
            persons.append(signatures)
        if is_all:
            if is_all_test:
                dataset = "bengali_test"
            else:
                dataset = "bengali"
            num_classes = DATASET_NUM_CLASSES[dataset]
            index = 0

    if dataset == "bengali" or dataset == "bengali_test":
        images = glob.glob(data_dir + "/bengali/*/*.tif")
        for person in range(num_classes):
            if is_genuine:
                num_of_signatures = DATASET_SIGNATURES_PER_PERSON["bengali_org"]
                index += 30
            else:
                num_of_signatures = DATASET_SIGNATURES_PER_PERSON["bengali_forg"]
            signatures = []
            for signature in range(num_of_signatures):
                signatures.append(images[index])
                index += 1
            if not is_genuine:
                index += 24
            persons.append(signatures)
        if is_all:
            if is_all_test:
                dataset = "dutch_test"
            else:
                dataset = "dutch"
            num_classes = DATASET_NUM_CLASSES[dataset]
            index = 0

    if dataset == "dutch" or dataset == "dutch_test":
        if is_genuine:
            for person in range(num_classes):
                images = glob.glob(
                    data_dir + "/dutch/genuine/" + str(person + 1) + "/*.PNG"
                )
                persons.append(images)

        else:
            for person in range(num_classes):
                images = glob.glob(
                    data_dir + "/dutch/forgery/" + str(person + 1) + "/*.PNG"
                )
                persons.append(images)
        if is_all:
            if is_all_test:
                dataset = "hindi_test"
            else:
                dataset = "hindi"
            num_classes = DATASET_NUM_CLASSES[dataset]
            index = 0

    if dataset == "hindi" or dataset == "hindi_test":
        images = glob.glob(data_dir + "/hindi/*/*.tif")
        for person in range(num_classes):
            if is_genuine:
                num_of_signatures = DATASET_SIGNATURES_PER_PERSON["hindi_org"]
                index += 30
            else:
                num_of_signatures = DATASET_SIGNATURES_PER_PERSON["hindi_forg"]
            signatures = []
            for signature in range(num_of_signatures):
                signatures.append(images[index])
                index += 1
            if not is_genuine:
                index += 24
            persons.append(signatures)
        if is_all:
            if is_all_test:
                dataset = "chinese_test"
            else:
                dataset = "chinese"
            num_classes = DATASET_NUM_CLASSES[dataset]
            index = 0

    if dataset == "chinese" or dataset == "chinese_test":
        if is_genuine:
            for person in range(num_classes):
                images = glob.glob(
                    data_dir + "/chinese/genuine/" + str(person + 1) + "/*.PNG"
                )
                persons.append(images)
        else:
            for person in range(num_classes):
                images = glob.glob(
                    data_dir + "/chinese/forgery/" + str(person + 1) + "/*.PNG"
                )
                persons.append(images)
        if is_all:
            if is_all_test:
                dataset = "gdps_test"
            else:
                dataset = "gdps"
            num_classes = DATASET_NUM_CLASSES[dataset]
            index = 0

    if dataset == "gdps" or dataset == "gdps_test":
        images = glob.glob(data_dir + "/GDPS/*/*.jpg")
        for person in range(num_classes):
            if is_genuine:
                num_of_signatures = DATASET_SIGNATURES_PER_PERSON["hindi_org"]
            else:
                index += 24
                num_of_signatures = DATASET_SIGNATURES_PER_PERSON["hindi_forg"]
            signatures = []
            for signature in range(num_of_signatures):
                signatures.append(images[index])
                index += 1
            if is_genuine:
                index += 30
            persons.append(signatures)
            if gdps_size is not None:
                if person == gdps_size - 1:
                    break
        if is_all:
            if is_all_test:
                dataset = "czech_test"
            else:
                dataset = "czech"
            num_classes = DATASET_NUM_CLASSES[dataset]
            index = 0

    if dataset == "czech" or dataset == "czech_test":
        if is_genuine:
            for person in range(num_classes):
                images = glob.glob(
                    data_dir + "/czech/genuine/" + str(person + 1) + "/*.jpg"
                )
                persons.append(images)

        else:
            for person in range(num_classes):
                images = glob.glob(
                    data_dir + "/czech/forgery/" + str(person + 1) + "/*.jpg"
                )
                persons.append(images)

    return persons


def convert_to_image(image_path: str, image_width: int = 150, image_height: int = 150):
    img = Image.open(image_path)
    img = img.resize((image_width, image_height))
    img = img.convert("L")
    img = img.point(lambda p: 255 if p > 210 else 0)  # Thresholding
    img = img.convert("1")  # udela to to co chci?? ANO
    img = np.array(img, dtype="float32")
    img = img[..., np.newaxis]
    return img


# Augmentations:
def rand_rotate(image):
    img = image.copy()
    w = img.shape[1]
    h = img.shape[0]
    if np.random.randint(0, 2) == 0:
        if np.random.randint(0, 2) == 0:
            angle = 10
        else:
            angle = -10
    else:
        if np.random.randint(0, 2) == 0:
            angle = 20
        else:
            angle = -20

    matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2), angle, 1.0
    )  # center cx, cy = w/2 h/2
    img = cv2.warpAffine(
        img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1)
    )
    return img


def rand_translation(image):
    img = image.copy()
    shift = 10.0
    width = img.shape[1]
    height = img.shape[0]
    direction = np.random.randint(0, 4)

    if direction == 0:  # UP
        matrix = np.float32([[1, 0, 0], [0, 1, -shift]])
    if direction == 1:  # DOWN
        matrix = np.float32([[1, 0, 0], [0, 1, shift]])
    if direction == 2:  # RIGHT
        matrix = np.float32([[1, 0, -shift], [0, 1, 0]])
    if direction == 3:  # LEFT
        matrix = np.float32([[1, 0, shift], [0, 1, 0]])
    img = cv2.warpAffine(
        img,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(1, 1, 1),
    )
    return img


def rand_zoom(image):
    img = image.copy()
    zoom = float(np.random.randint(8, 12)) / 10
    cy, cx = [i / 2 for i in img.shape[:-1]]
    matrix = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
    img = cv2.warpAffine(
        img,
        matrix,
        img.shape[1::-1],
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(1, 1, 1),
    )
    return img


def rand_shear(image):
    img = image.copy()
    axe = np.random.randint(0, 2)
    width = img.shape[1]
    height = img.shape[0]
    if axe == 0:
        matrix = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    if axe == 1:
        matrix = np.float32([[1, 0, 0], [0.5, 1, 0], [0, 0, 1]])

    img = cv2.warpPerspective(
        img,
        matrix,
        (int(width * 1.2), int(height * 1.2)),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(1, 1, 1),
    )

    return img


def rand_noise(image):
    img = image.copy()
    gaussian_noise = np.random.normal(0, 0.3, img.shape)
    img += gaussian_noise
    img = np.clip(img, 0, 1)
    return img


# Augmentator
def augment_image(image):
    augmented_images = []

    rotated_image = rand_rotate(image)
    rotated_image = rotated_image[..., np.newaxis]
    augmented_images.append(rotated_image)

    sheared_image = rand_shear(image)
    sheared_image = cv2.resize(sheared_image, (image.shape[0], image.shape[1]))
    sheared_image = sheared_image[..., np.newaxis]
    augmented_images.append(sheared_image)

    resized_image = rand_zoom(image)
    resized_image = resized_image[..., np.newaxis]
    augmented_images.append(resized_image)

    shifted_image = rand_translation(image)
    shifted_image = shifted_image[..., np.newaxis]
    augmented_images.append(shifted_image)

    noisy_image = rand_noise(image)
    augmented_images.append(noisy_image)

    return augmented_images


def convert_array_to_image_labels(
        image_path_array,
        image_width: int = 150,
        image_height: int = 150,
        augmented: bool = False,
        genuine: bool = False,
        size: int = None,
):
    labels = []
    image_array = []
    index = 0
    for person in image_path_array:
        if size:
            if index > size:
                break
            else:
                index += 1
        for img in person:
            img = convert_to_image(img, image_width=image_width, image_height=image_height)
            image_array.append(img)
            labels.append(1 if genuine else 0)
            if augmented:
                augmented_images = augment_image(img)
                image_array.extend(augmented_images)
                # augmented_images.insert(0,img)
                # augmented_labels = ['Original', 'Rotate', 'Shear', 'Zoom', 'Transition', 'Gaussian noise']
                # plot_images(augmented_images,  num_column=6,
                #            title='Augmented images') #THIS ONLY FOR SHOWING PURPOSES
                if genuine:
                    labels.extend([1 for i in range(len(augmented_images))])
                else:
                    labels.extend([0 for i in range(len(augmented_images))])
    if size and size < len(labels):
        image_sized_array = []
        label_sized_array = []
        rng = np.random.default_rng()
        indieces = rng.choice(len(image_array), size=size, replace=False, shuffle=True)
        for i in indieces:
            image = image_array[i]
            image_sized_array.append(image)
            label_sized_array.append(labels[i])
        return image_sized_array, label_sized_array
    return image_array, labels


def combine_orig_forg(orig_data, forg_data, orig_labels, forg_labels, shuffle: bool = True):
    data = orig_data + forg_data
    labels = orig_labels + forg_labels
    if shuffle:
        data, labels = sklearn.utils.shuffle(data, labels, random_state=42)

    return data, labels


# CNN Loader
def loader_for_cnn(
        data_dir: str = "data",
        image_width: int = 150,
        image_height: int = 150,
        dataset: str = "cedar",
        augmented: str = False,
        size: int = None,
        shuffle: bool = True,
):
    if size:
        size /= 2
        size = int(size)

    start_time = time.time()

    orig_data = create_data(data_dir, dataset=dataset, is_genuine=True)
    forg_data = create_data(data_dir, dataset=dataset, is_genuine=False)
    print(f"Genuine DATA: {len(orig_data)}")
    print(f"Forgery DATA: {len(forg_data)}")
    orig_data, orig_labels = convert_array_to_image_labels(
        orig_data,
        image_width=image_width,
        image_height=image_height,
        genuine=True,
        augmented=augmented,
        size=size,
    )
    forg_data, forg_labels = convert_array_to_image_labels(
        forg_data,
        image_width=image_width,
        image_height=image_height,
        genuine=False,
        augmented=augmented,
        size=size,
    )
    data, labels = combine_orig_forg(
        orig_data, forg_data, orig_labels, forg_labels, shuffle=shuffle
    )
    print(f"Dataset: {len(data)} and labels: {len(labels)}")

    data, labels = np.array(data), np.array(labels, dtype=np.float32)

    end_time = time.time()
    print(f"It took {(end_time - start_time):.2f}")
    return data, labels


# SNN LOADER
def convert_pairs_to_image_pairs(
        pair_array, labels, image_width: int = 150, image_height: int = 150, output_size: int = 0,
        augmented: bool = False
):
    image_pair_array = []
    new_labels = []
    index = 0

    if output_size == 0 or output_size > len(pair_array):
        output_size = len(pair_array)
        for pair in pair_array:
            if index == output_size:
                break
            image1 = convert_to_image(pair[0], image_width=image_width, image_height=image_height)
            image2 = convert_to_image(pair[1], image_width=image_width, image_height=image_height)
            image_pair_array.append((image1, image2))
            new_labels.append(labels[index])
            if augmented:
                rng = np.random.default_rng()
                augmented_img1 = augment_image(image1)
                augmented_img2 = augment_image(image2)
                indices1 = rng.choice(
                    len(augmented_img1),
                    size=len(augmented_img1),
                    replace=False,
                    shuffle=True,
                )
                indices2 = rng.choice(
                    len(augmented_img2),
                    size=len(augmented_img2),
                    replace=False,
                    shuffle=True,
                )
                for i in range(len(augmented_img1)):
                    new_img1 = augmented_img1[indices1[i]]
                    new_img2 = augmented_img2[indices2[i]]
                    image_pair_array.append((new_img1, new_img2))
                    new_labels.append(labels[index])
            index += 1
        return image_pair_array, new_labels
    rng = np.random.default_rng()
    indieces = rng.choice(
        len(pair_array), size=output_size, replace=False, shuffle=True
    )
    for i in indieces:
        image1 = convert_to_image(pair_array[i][0], image_width=image_width, image_height=image_height)
        image2 = convert_to_image(pair_array[i][1], image_width=image_width, image_height=image_height)
        image_pair_array.append((image1, image2))
        new_labels.append(labels[i])
        if augmented:
            rng = np.random.default_rng()
            augmented_img1 = augment_image(image1)
            augmented_img2 = augment_image(image2)
            indices1 = rng.choice(
                len(augmented_img1),
                size=len(augmented_img1),
                replace=False,
                shuffle=True,
            )
            indices2 = rng.choice(
                len(augmented_img2),
                size=len(augmented_img2),
                replace=False,
                shuffle=True,
            )
            for j in range(len(augmented_img1)):
                new_img1 = augmented_img1[indices1[j]]
                new_img2 = augmented_img2[indices2[j]]
                image_pair_array.append((new_img1, new_img2))
                new_labels.append(labels[i])

    del pair_array, labels
    return image_pair_array, new_labels


def make_pairs(orig_data, forg_data):
    orig_pairs, forg_pairs = [], []

    # if output_size == 0:
    for orig, forg in zip(orig_data, forg_data):
        orig_pairs.extend(list(itertools.combinations(orig, 2)))
        for i in range(len(forg)):
            forg_pairs.extend(
                list(itertools.product(orig[i: i + 1], random.sample(forg, 12)))
            )
    data_pairs = orig_pairs + forg_pairs
    orig_pair_labels = [1] * len(orig_pairs)
    orig_forg_labels = [0] * len(forg_pairs)

    label_pairs = orig_pair_labels + orig_forg_labels
    del orig_data, forg_data
    print(f"Len of genuine pairs: {len(orig_pairs)}")
    print(f"Len of forgery pairs {len(forg_pairs)}")
    del orig_pairs, forg_pairs
    print(f"Len of final pairs: {len(data_pairs)}")
    return data_pairs, label_pairs


def loader_for_snn(
        data_dir: str = "data",
        image_width: int = 150,
        image_height: int = 150,
        dataset: str = "cedar",
        augmented: bool = False,
        size: int = 0,
        gdps_size: int = None,
):
    if augmented and size != 0:
        size /= 6
        size = int(size)

    start_time = time.time()

    orig_data = create_data(
        data_dir, dataset=dataset, is_genuine=True, gdps_size=gdps_size
    )
    forg_data = create_data(
        data_dir, dataset=dataset, is_genuine=False, gdps_size=gdps_size
    )

    print("___________________Creating pairs__________________")
    data_pairs, data_labels = make_pairs(orig_data, forg_data)
    print("___________________Loading images__________________")
    data_pairs, data_labels = convert_pairs_to_image_pairs(
        data_pairs,
        data_labels,
        image_width=image_width,
        image_height=image_height,
        output_size=size,
        augmented=augmented,
    )
    print("_____________________Done__________________________________")
    end_time = time.time()
    print(f"It took : {(end_time - start_time):.2f}")
    print(
        f"Created Data: {len(data_pairs)} , labels: {len(data_labels)} with data shape = {data_pairs[0][0].shape}"
    )
    # visualize_snn_pair_sample(data_pairs, data_labels, title='Data pairs', numer_of_samples=5)
    data_pairs, data_labels = np.array(data_pairs), np.array(
        data_labels, dtype=np.float32
    )
    return data_pairs, data_labels
