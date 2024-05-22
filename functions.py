import matplotlib as mpl

mpl.use("TkAgg")
import random

import matplotlib.pyplot as plt
import numpy as np
import shap
import tensorflow as tf

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
import cv2
import pywt
from keras import backend as K
from keras import utils
from keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


# Visualizations
def show_single_image(img):
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show(block=True)


def plot_images(
        image_array,
        image_array_label=[],
        num_column: int = 5,
        title: str = "Images in dataset",
        save_path: str = None,
):
    fig, axes = plt.subplots(1, num_column, figsize=(20, 20))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()
    index = 0
    for img, ax in zip(image_array, axes):
        ax.imshow(img, cmap="Greys_r")
        if image_array_label != []:
            ax.set_title(image_array_label[index])
            index += 1
        # ax.axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_path is None:
        plt.show(block=True)
    else:
        plt.savefig(save_path)


def plot_training(hist):
    fig = plt.figure(figsize=(7, 7))
    plt.plot(hist.history["loss"], color="teal", label="loss")
    plt.plot(hist.history["val_loss"], color="orange", label="val_loss")
    fig.suptitle("Loss", fontsize=20)
    plt.legend(loc="upper left")
    plt.show(block=True)

    fig = plt.figure(figsize=(7, 7))
    plt.plot(hist.history["accuracy"], color="teal", label="accuracy")
    plt.plot(hist.history["val_accuracy"], color="orange", label="val_accuracy")
    fig.suptitle("Accuracy", fontsize=20)
    plt.legend(loc="upper left")
    plt.show(block=True)


def visualize_snn_sample_signature_for_signer(
        orig_data, forg_data, image_width: int = 150, image_height: int = 150
):
    k = np.random.randint(len(orig_data))
    orig_data_signature = random.sample(orig_data[k], 2)
    forg_data_signature = random.sample(forg_data[k], 1)
    print(orig_data_signature[0])
    print(orig_data_signature[1])
    print(forg_data_signature[0])
    orig_im1 = cv2.imread(orig_data_signature[0], 0)
    orig_im1 = cv2.resize(orig_im1, (image_width, image_height))
    orig_im2 = cv2.imread(orig_data_signature[1], 0)
    orig_im2 = cv2.resize(orig_im2, (image_width, image_height))
    forg_im = cv2.imread(forg_data_signature[0], 0)
    forg_im = cv2.resize(forg_im, (image_width, image_height))
    img_array_to_show = [orig_im1, orig_im2, forg_im]
    img_array_label = ["genuine", "genuine", "forgery"]
    plot_images(img_array_to_show, img_array_label, num_column=len(img_array_to_show))


def show_pair(pairs, labels, title: str = "Image pairs", columns: int = 2, rows: int = 1):
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 5, rows * 2))
    fig.suptitle(title)
    if rows == 1:
        axes[0].imshow(pairs[0][0], cmap="gray")
        axes[0].set_title(labels[0][0])
        axes[0].axis("off")
        axes[1].imshow(pairs[0][1], cmap="gray")
        axes[1].set_title(labels[0][1])
        axes[1].axis("off")
    else:
        for row in range(rows):
            img_pair = pairs[row]
            label = labels[row]
            for column in range(columns):
                axes[row, column].imshow(img_pair[column], cmap="gray")
                axes[row, column].set_title(label[column])
                axes[row, column].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show(block=True)


def visualize_snn_pair_sample(
        pair_array, label_array, title: str = "Pair sample", number_of_samples: int = 5
):
    pairs = []
    label = []
    for i in range(number_of_samples):
        k = np.random.randint(0, len(pair_array))
        img1 = pair_array[k][0]
        img2 = pair_array[k][1]
        pairs.append([img1, img2])
        label.append(["Genuine", "Genuine" if label_array[k] == 1 else "Forgery"])

    show_pair(pairs, label, title=title, columns=2, rows=number_of_samples)


# Euclidan distance
@tf.function
def euclidan_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def euclidan_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def EarlyStopping():
    stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, mode="min", min_delta=0.0001, verbose=1
    )
    return stopper


def callbacks_stop_checkpoint():
    callbacks = [
        EarlyStopping(patience=12, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
        ModelCheckpoint(
            "./models/SNN_CEDAR-{epoch:03d}.h5", verbose=12, save_weights_only=True
        ),
    ]
    return callbacks


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def CSVLogger(filename:str):
    logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
    return logger


def callbacks_schedule_lr(filename:str):
    callback = [
        LearningRateScheduler(scheduler, verbose=10),
        CSVLogger(filename),
        EarlyStopping(),
    ]
    return callback


def get_image_strokes(img):
    _, inverted_image = cv2.threshold(img, 0.5, 1, cv2.THRESH_BINARY_INV)
    inverted_image = np.array(inverted_image, dtype=np.uint8)
    contours, _ = cv2.findContours(
        inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    strokes = len(contours)
    return strokes


def image_for_local(image, size: int = 15):
    small_image_array = []
    for row in range(0, image.shape[0], size):
        for col in range(0, image.shape[1], size):
            small_image = image[row: row + size, col: col + size]
            small_image_array.append(small_image)
    small_image_array = np.array(small_image_array, dtype="float32")
    return small_image_array


def wavelet_transformation(image):
    coeffs = pywt.dwt2(data=image, wavelet="haar")
    cA, (cH, cV, cD) = coeffs

    # normalized
    cA = (cA - cA.min()) / (cA.max() - cA.min())
    cH = (cH - cH.min()) / (cH.max() - cH.min())
    cA = cA.flatten()
    cH = cH.flatten()
    wavelet_features = np.concatenate((cA, cH))
    wavelet_features = wavelet_features.flatten()
    return wavelet_features


def plot_non_white_pixels(non_white_pixels_rows, non_white_pixels_columns):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(non_white_pixels_rows, range(len(non_white_pixels_rows)), color="blue")
    plt.title("Non-white Pixels in Rows")
    plt.xlabel("Count")
    plt.ylabel("Row Index")
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.plot(
        range(len(non_white_pixels_columns)), non_white_pixels_columns, color="red"
    )
    plt.title("Non-white Pixels in Columns")
    plt.xlabel("Column Index")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show(block=True)


def count_none_white_pixels(image, count_axis: bool = True, show: bool = False):
    pixel_count = np.sum(image == 0)
    if count_axis:
        non_white_pixels_rows = np.sum(image == 0, axis=1)
        non_white_pixels_columns = np.sum(image == 0, axis=0)
        if show:
            plot_non_white_pixels(non_white_pixels_rows, non_white_pixels_columns)
        return pixel_count, non_white_pixels_rows, non_white_pixels_columns
    return pixel_count


# Center of mass, Normalized area ,Aspect Ratio, Tri surface feature,six-fold surface feature and Transition feature
# https://www.researchgate.net/publication/258650160_Handwritten_Signature_Verification_using_Neural_Network
def calculate_center_of_mass(image):
    width = image.shape[1]
    height = image.shape[0]
    half_width = width // 2
    first_half = image[:half_width, :]
    second_half = image[half_width:, :]

    M1 = cv2.moments(first_half)
    M2 = cv2.moments(second_half)

    center_of_mass_first_half_x = M1["m10"] / M1["m00"] if M1["m00"] != 0 else 0
    center_of_mass_first_half_y = M1["m01"] / M1["m00"] if M1["m00"] != 0 else 0
    center_of_mass_second_half_x = M2["m10"] / M2["m00"] if M2["m00"] != 0 else 0
    center_of_mass_second_half_y = M2["m01"] / M2["m00"] if M2["m00"] != 0 else 0

    # normalization
    center_of_mass_first_half_normalized = np.array(
        [center_of_mass_first_half_x / half_width, center_of_mass_first_half_y / height]
    )
    center_of_mass_second_half_normalized = np.array(
        [
            center_of_mass_second_half_x / half_width,
            center_of_mass_second_half_y / height,
        ]
    )

    center_of_mass_normalized = np.concatenate(
        [center_of_mass_first_half_normalized, center_of_mass_second_half_normalized]
    )

    # return center_of_mass_first_half_normalized, center_of_mass_second_half_normalized
    return center_of_mass_normalized


def calculate_normalized_shape(image):
    image = image[:, :, 0]
    img_area = np.sum(image == 0)
    pixel_indices = np.where(image == 0)
    rows, cols = pixel_indices
    if len(rows) == 0 or len(cols) == 0:
        return 0
    bounding_box_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    normalized_shape = img_area / bounding_box_area
    return normalized_shape


def calculate_aspect_ratio(image):
    image = image[:, :, 0]
    pixel_indices = np.where(image == 0)
    rows, cols = pixel_indices
    if len(rows) == 0 or len(cols) == 0:
        return 0
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    aspect_ratio = width / height

    return aspect_ratio


def calculate_tri_surface_area(image):
    tri_width = image.shape[1] // 3
    parts = [image[:, i * tri_width: (i + 1) * tri_width] for i in range(3)]
    areas = [calculate_normalized_shape(part) for part in parts]
    return areas


def six_fold_surface(image):
    image = image[:, :, 0]
    part_width = image.shape[1] // 3
    parts = [image[:, i * part_width: (i + 1) * part_width] for i in range(3)]
    all_features = []
    for part in parts:
        features = []
        pixel_indieces = np.where(part == 0)
        rows, cols = pixel_indieces
        if len(rows) == 0 or len(rows) == 0:
            features.append([0, 0])
            features.append([0, 0])
            features.append([0, 0])
            all_features.append(features)
        else:
            boundingbox_width = [min(cols), max(cols)]
            boundingbox_height = [min(rows), max(rows)]
            bounding_box = [
                boundingbox_width[1] - boundingbox_width[0] + 1,
                boundingbox_height[1] - boundingbox_height[0] + 1,
            ]
            Mx = cv2.moments(
                part[
                boundingbox_height[0]: boundingbox_height[1],
                boundingbox_width[0]: boundingbox_width[1],
                ]
            )
            center_of_mass_x = Mx["m10"] / Mx["m00"] if Mx["m00"] != 0 else 0
            center_of_mass_y = Mx["m01"] / Mx["m00"] if Mx["m00"] != 0 else 0
            area_above_center = np.sum(
                part[
                boundingbox_height[0]: (
                        boundingbox_height[0] + int(center_of_mass_y)
                ),
                boundingbox_width[0]: boundingbox_width[1],
                ]
                == 0
            )
            area_bellow_center = np.sum(
                part[
                (
                        boundingbox_height[0] + int(center_of_mass_y)
                ): boundingbox_height[1],
                boundingbox_width[0]: boundingbox_width[1],
                ]
                == 0
            )
            area_above_center /= (
                                         boundingbox_height[0] + int(center_of_mass_y)
                                 ) * bounding_box[0]
            area_bellow_center /= (
                                          part.shape[0] - (boundingbox_height[0] + int(center_of_mass_y))
                                  ) * bounding_box[0]
            center_of_mass_x = center_of_mass_x / (bounding_box[0])  # normalized
            center_of_mass_y = center_of_mass_y / (bounding_box[1])
            bounding_box[0] /= part.shape[1]  # normalized
            bounding_box[1] /= part.shape[0]
            features.append(bounding_box)
            features.append([center_of_mass_x, center_of_mass_y])

            features.append([area_bellow_center, area_above_center])
            all_features.append(features)

    return all_features


def visualize_with_shap(data, model, pred, save_path: str = None):
    shap.initjs()
    masker = shap.maskers.Image("inpaint_telea", data[0].shape)
    explainer = shap.Explainer(model, masker)
    shap_values = explainer(data, outputs=shap.Explanation.argsort.flip[:1])
    labels = pred
    labels = np.array(labels)
    if save_path:
        fig = shap.image_plot(shap_values, labels=labels, show=False)
        plt.savefig(save_path)
        print("fig saved")
    else:
        shap.image_plot(shap_values, labels=labels)


def shuffle_data(data, labels):
    data_shuffled_array = []
    label_shuffled_array = []
    rng = np.random.default_rng()
    indices = rng.choice(len(data), size=len(data), replace=False, shuffle=True)
    for i in indices:
        image = data[i]
        data_shuffled_array.append(image)
        label_shuffled_array.append(labels[i])
    data_shuffled_array = np.array(data_shuffled_array, dtype=np.float32)
    label_shuffled_array = np.array(label_shuffled_array, dtype=np.float32)
    return data_shuffled_array, label_shuffled_array


def prediction_to_label(predictions):
    labels = []
    if type(predictions) is float:
        if np.isnan(predictions):
            labels.append(["Could not predict"])
        if predictions > 0.5:
            labels.append(["Genuine"])
        else:
            labels.append(["Forgery"])
    else:
        for prediction in predictions:
            if np.isnan(prediction):
                labels.append(["Could not predict"])
                continue
            if prediction > 0.5:
                labels.append(["Genuine"])
            else:
                labels.append(["Forgery"])
    return np.array(labels)


def overlay_heatmap(image, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    # Resize the heatmap to match the size of the input image
    heatmap_resized = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    # Handle NaN
    heatmap_resized = np.nan_to_num(heatmap_resized)
    # Scale the heatmap to the range of 0 to 255
    heatmap_rescaled = (heatmap_resized * 255).astype(np.uint8)

    # Apply colormap to the heatmap
    heatmap_colored = cv2.applyColorMap(heatmap_rescaled, colormap)

    # Convert the original image to the range of 0 to 255
    image_uint8 = (image * 255).astype(np.uint8)

    # Overlay the heatmap onto the original image
    overlaid_image = cv2.addWeighted(
        cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR),
        alpha,
        heatmap_colored,
        1 - alpha,
        0,
    )

    return overlaid_image


def add_features(data, is_pair: bool = True, feature_type: str = "strokes"):
    feature = []
    if feature_type == "strokes":
        if is_pair:
            for pair in data:
                stroke1 = get_image_strokes(pair[0])
                stroke1 /= 1000
                stroke2 = get_image_strokes(pair[1])
                stroke2 /= 1000
                feature.append([stroke1, stroke2])
        else:
            for img in data:
                stroke1 = get_image_strokes(img)
                stroke1 /= 1000
                feature.append(stroke1)
    elif feature_type == "histogram":
        if is_pair:
            for pair in data:
                pixel1, horizontal1, vertical1 = count_none_white_pixels(
                    pair[0], count_axis=True
                )
                horizontal1 = horizontal1.flatten()
                vertical1 = vertical1.flatten()
                histogram1 = np.concatenate([horizontal1, vertical1])
                histogram1 = np.concatenate([[pixel1], histogram1])
                pixel2, horizontal2, vertical2 = count_none_white_pixels(
                    pair[1], count_axis=True
                )
                horizontal2 = horizontal2.flatten()
                vertical2 = vertical2.flatten()
                histogram2 = np.concatenate([horizontal2, vertical2])
                histogram2 = np.concatenate([[pixel2], histogram2])
                feature.append([histogram1, histogram2])
        else:
            for img in data:
                pixel, horizontal, vertical = count_none_white_pixels(
                    img, count_axis=True
                )
                horizontal = horizontal.flatten()
                vertical = vertical.flatten()
                histogram = np.concatenate([horizontal, vertical])
                histogram = np.concatenate([[pixel], histogram])
                feature.append(histogram)

    elif feature_type == "wavelet":
        if is_pair:
            for pair in data:
                wavelet1 = wavelet_transformation(pair[0])
                wavelet2 = wavelet_transformation(pair[1])
                feature.append([wavelet1, wavelet2])
        else:
            for img in data:
                wavelet = wavelet_transformation(img)
                feature.append(wavelet)
    elif feature_type == "tri_shape":
        if is_pair:
            for pair in data:
                mass1 = calculate_center_of_mass(pair[0])
                mass2 = calculate_center_of_mass(pair[1])
                norm1 = calculate_normalized_shape(pair[0])
                norm2 = calculate_normalized_shape(pair[1])
                aspect1 = calculate_aspect_ratio(pair[0])
                aspect2 = calculate_aspect_ratio(pair[1])
                # print(f"mass1: {mass1}, norm1: {norm1}, aspect1: {aspect1} ")
                # print(f"mass2: {mass2}, norm1: {norm2}, aspect2: {aspect2} ")
                feature.append(
                    [
                        [mass1[0], mass1[1], mass1[2], mass1[3], norm1, aspect1],
                        [mass2[0], mass2[1], mass2[2], mass2[3], norm2, aspect2],
                    ]
                )
        else:
            for img in data:
                mass = calculate_center_of_mass(img)
                norm = calculate_normalized_shape(img)
                aspect = calculate_aspect_ratio(img)
                feature.append([mass[0], mass[1], mass[2], mass[3], norm, aspect])
    elif feature_type == "tri_surface":
        if is_pair:
            for pair in data:
                tri_surface1 = calculate_tri_surface_area(pair[0])
                tri_surface2 = calculate_tri_surface_area(pair[1])
                feature.append(
                    [
                        [tri_surface1[0], tri_surface1[1], tri_surface1[2]],
                        [tri_surface2[0], tri_surface2[1], tri_surface2[2]],
                    ]
                )
        else:
            for img in data:
                tri_surface = calculate_tri_surface_area(img)
                feature.append([tri_surface[0], tri_surface[1], tri_surface[2]])
    elif feature_type == "six_fold":
        if is_pair:
            for pair in data:
                six_fold1 = six_fold_surface(pair[0])
                six_fold2 = six_fold_surface(pair[1])
                six_fold1 = np.concatenate(
                    (
                        six_fold1[0][0],
                        six_fold1[0][1],
                        six_fold1[0][2],
                        six_fold1[1][0],
                        six_fold1[1][1],
                        six_fold1[1][2],
                        six_fold1[2][0],
                        six_fold1[2][1],
                        six_fold1[2][2],
                    )
                )
                six_fold2 = np.concatenate(
                    (
                        six_fold2[0][0],
                        six_fold2[0][1],
                        six_fold2[0][2],
                        six_fold2[1][0],
                        six_fold2[0][1],
                        six_fold2[1][2],
                        six_fold2[2][0],
                        six_fold2[2][1],
                        six_fold2[2][2],
                    )
                )
                feature.append([six_fold1, six_fold2])
        else:
            for img in data:
                six_fold = six_fold_surface(img)
                six_fold = np.concatenate(
                    (
                        six_fold[0][0],
                        six_fold[0][1],
                        six_fold[0][2],
                        six_fold[1][0],
                        six_fold[1][1],
                        six_fold[1][2],
                        six_fold[2][0],
                        six_fold[2][1],
                        six_fold[2][2],
                    )
                )
                feature.append(six_fold)
    elif feature_type == "local" or feature_type == "local_solo":
        if is_pair:
            for pair in data:
                local1 = image_for_local(pair[0])
                local2 = image_for_local(pair[1])
                feature.append([local1, local2])
        else:
            for img in data:
                local = image_for_local(img)
                feature.append(local)

    feature = np.array(feature)
    print(f"feature shape: {feature.shape}")
    return feature
