from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf

import functions

tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

import loader
import model
from conf import DATASETS, DATASETS_TEST, FEATURES, MODEL_DIR
from validator import model_evaluation, model_prediction

# IMG_HEIGHT = 150
# IMG_WIDTH = 150
CHANNELS = 1

image_shape = (None, 100, 100, 3)


def cnn_train(
        epochs: int = 100,
        batch_size: int = 32,
        img_width: int = 150,
        img_height: int = 150,
        dataset: str = "cedar",
        augmented: bool = False,
        save_name: str = None,
        size: int = None,
        feature_type: str = None,
        show_fig: bool = False,
):
    if save_name is None:
        csv_name = "CNN"
    else:
        csv_name = save_name
    if size == 0:
        size = None

    data, labels = loader.loader_for_cnn(
        image_width=img_width,
        image_height=img_height,
        dataset=dataset,
        augmented=augmented,
        size=size,
    )
    if feature_type is not None:
        feature = functions.add_features(data, is_pair=False, feature_type=feature_type)
        if feature_type == "wavelet" or feature_type == "histogram":
            CNNMODEL = model.cnn_feature_model(
                image_shape=(img_width, img_height, CHANNELS),
                feature_type=feature_type,
                feature_shape=feature.shape[1],
            )
        elif feature_type == "local" or feature_type == "local_solo":
            CNNMODEL = model.cnn_feature_model(
                image_shape=(img_width, img_height, CHANNELS),
                feature_type=feature_type,
                feature_shape=(
                    feature.shape[1],
                    feature.shape[2],
                    feature.shape[3],
                    feature.shape[4],
                ),
            )
        else:
            CNNMODEL = model.cnn_feature_model(
                image_shape=(img_width, img_height, CHANNELS), feature_type=feature_type
            )
    else:
        CNNMODEL = model.cnn_model(image_shape=(img_width, img_height, CHANNELS))

    if feature_type is not None and feature_type != "local_solo":
        hist = CNNMODEL.fit(
            x=([data[:, ], feature[:, ]]),
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            shuffle=True,
            callbacks=functions.callbacks_schedule_lr((csv_name + ".csv")),
        )
    elif feature_type == "local_solo":
        hist = CNNMODEL.fit(
            x=feature,
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            shuffle=True,
            callbacks=functions.callbacks_schedule_lr((csv_name + ".csv")),
        )
    else:
        hist = CNNMODEL.fit(
            x=data,
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            shuffle=True,
            callbacks=functions.callbacks_schedule_lr((csv_name + ".csv")),
        )
    # CNNMODEL.summary()

    if save_name is not None:
        print("Model saved")
        CNNMODEL.save(os.path.join("models", (save_name + ".h5")))
    if show_fig:
        functions.plot_training(hist)


def snn_train(
        epochs: int = 100,
        batch_size: int = 32,
        img_width: int = 150,
        img_height: int = 150,
        dataset: str = "cedar",
        augmented: bool = False,
        size: int = 2000,
        feature_type: str = None,
        gdps_size: int = None,
        save_name: str = None,
        show_fig: bool = False,
):
    if save_name is None:
        csv_name = "SNN"
    else:
        csv_name = save_name
    data_pairs, data_labels = loader.loader_for_snn(
        image_width=img_width,
        image_height=img_height,
        dataset=dataset,
        augmented=augmented,
        size=size,
        gdps_size=gdps_size,
    )
    if feature_type is not None:
        feature = functions.add_features(data_pairs, feature_type=feature_type)
        if feature_type == "wavelet" or feature_type == "histogram":
            SNNMODEL = model.snn_model(
                image_shape=(img_width, img_height, CHANNELS),
                feature_shape=feature.shape[2],
                feature_type=feature_type,
            )
        elif feature_type == "local" or feature_type == "local_solo":
            SNNMODEL = model.snn_model(
                image_shape=(img_width, img_height, CHANNELS),
                feature_type=feature_type,
                feature_shape=(
                    feature.shape[2],
                    feature.shape[3],
                    feature.shape[4],
                    feature.shape[5],
                ),
            )
        else:
            SNNMODEL = model.snn_model(
                image_shape=(img_width, img_height, CHANNELS), feature_type=feature_type
            )
    else:
        SNNMODEL = model.snn_model(image_shape=(img_width, img_height, CHANNELS))

    if feature_type is not None and feature_type != "local_solo":
        hist = SNNMODEL.fit(
            x=(
                [
                    data_pairs[:, 0, :, :],
                    feature[:, 0],
                    data_pairs[:, 1, :, :],
                    feature[:, 1],
                ]
            ),
            y=data_labels,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_split=0.2,
            callbacks=functions.callbacks_schedule_lr((csv_name + ".csv")),
        )
    elif feature_type == "local_solo":
        hist = SNNMODEL.fit(
            x=([feature[:, 0], feature[:, 1]]),
            y=data_labels,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_split=0.2,
            callbacks=functions.callbacks_schedule_lr((csv_name + ".csv")),
        )
    else:
        hist = SNNMODEL.fit(
            x=([data_pairs[:, 0, :, :], data_pairs[:, 1, :, :]]),
            y=data_labels,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_split=0.2,
            callbacks=functions.callbacks_schedule_lr((csv_name + ".csv")),
        )
    # SNN.summary()

    if save_name is not None:
        print("Model saved")
        SNNMODEL.save(os.path.join("models", (save_name + ".h5")))

    if show_fig:
        functions.plot_training(hist)


def choose_dataset(dataset_type: str):
    if dataset_type == "test":
        datasets = DATASETS_TEST
    else:
        datasets = DATASETS
    for i in range(len(datasets)):
        print(f"{i}: {datasets[i]}")
    dataset = int(input("Chose dataset: "))
    return datasets[dataset]


def handle_train():
    epochs = int(input("Number of epochs: "))
    batch_size = int(input("Batch size: "))
    width = 150
    height = 150
    augmented = bool(input("Augmented data (True, ENTER): "))

    ans = int(
        input(
            "Do you wanna activate CNN(0) or SNN(1) or Package CNN(2) or  Package SNN(3):  "
        )
    )
    if ans == 0 or ans == 1:
        save = input("Name of model to save: ")
        if save == "":
            save = None
        dataset = choose_dataset("train")
        for i in range(len(FEATURES)):
            print(f"{i}: {FEATURES[i]}")
        feature = int(input("Chose feature: "))
        if feature == 0:
            feature_type = None
        else:
            feature_type = FEATURES[feature]

    if ans == 0:
        size = int(input("Size of data: "))
        cnn_train(
            epochs=epochs,
            batch_size=batch_size,
            img_width=width,
            img_height=height,
            augmented=augmented,
            dataset=dataset,
            size=size,
            save_name=save,
            feature_type=feature_type,
        )
    elif ans == 1:
        size = int(input("Size of pairs: "))

        snn_train(
            epochs=epochs,
            batch_size=batch_size,
            img_width=width,
            img_height=height,
            dataset=dataset,
            size=size,
            feature_type=feature_type,
            save_name=save,
        )

    elif ans == 2 or ans == 3:
        model_dir = input("Enter output dir: ")
        size = int(input("Size of data: "))
        for dataset in DATASETS:
            for feature in FEATURES:
                if feature == "None":
                    feature = None
                if ans == 2:
                    save = model_dir + "/" + "CNN_" + dataset + "_" + str(feature)
                    print(f"Training for CNN with {dataset} and {feature}")
                    cnn_train(
                        epochs=epochs,
                        batch_size=batch_size,
                        img_width=width,
                        img_height=height,
                        augmented=augmented,
                        dataset=dataset,
                        size=size,
                        save_name=save,
                        feature_type=feature,
                    )
                else:
                    save = model_dir + "/" + "SNN_" + dataset + "_" + str(feature)
                    print(f"Training for SNN with {dataset} and {feature}")
                    snn_train(
                        epochs=epochs,
                        batch_size=batch_size,
                        img_width=width,
                        img_height=height,
                        dataset=dataset,
                        augmented=augmented,
                        size=size,
                        feature_type=feature,
                        save_name=save,
                    )
    else:
        return


def handle_evaluate():
    data_dir = input("Enter data directory (or press ENTER): ")
    if data_dir == "":
        data_dir = "test"
    dataset = choose_dataset("test")
    model_path = input("Enter path to model (or press ENTER): ")
    if model_path == "":
        model_dir = input("Enter directory with models (or press ENTER): ")
        if model_dir == "":
            model_dir = MODEL_DIR
        model_evaluation(data_dir, dataset, model_dir=model_dir)
    else:
        model_evaluation(data_dir, dataset, model_path=model_path)


def load_prediction_data(method: int):
    image_array = []
    size = int(input("Enter data size for prediction: "))
    for i in range(size):
        if method == 1:
            image_path = input("Enter path to image: ")
            image_array.append(image_path)
        if method == 2:
            pair1_path = input("Enter path to reference image: ")
            pair2_path = input("Enter path to image for prediction: ")
            image_array.append([pair1_path, pair2_path])
    return image_array


def handle_predict():
    method = -1
    while method != 0:
        method = int(input("Would you like to predict by CNN(1) or SNN(2), exit(0): "))
        if method == 0:
            break
        image_array = load_prediction_data(method)
        model_path = input("Enter path to model (or press ENTER): ")
        if model_path == "":
            model_dir = input("Enter directory with models (or press ENTER): ")
            if model_dir == "":
                model_dir = MODEL_DIR
            model_prediction(image_array, method=method, model_dir=model_dir)
        else:
            model_prediction(image_array, method=method, model_path=model_path)


def main():
    ans = -1
    while ans != 0:
        ans = int(
            input(
                "Would you like to train model(1), evaluate model(2), get prediction(3) or end program(0): "
            )
        )
        if ans == 1:
            handle_train()
        elif ans == 2:
            handle_evaluate()
        elif ans == 3:
            handle_predict()


if __name__ == "__main__":
    main()
