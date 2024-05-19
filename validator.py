import glob

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

import functions
import loader
from conf import FEATURES
from functions import overlay_heatmap, prediction_to_label, visualize_with_shap
from model import make_gradcam_heatmap


def get_model(model_dir: str):
    models = glob.glob(model_dir + "/*.h5")
    print("Available models: ")
    for i in range(len(models)):
        print(f"{i}: {models[i]}")
    index = int(input("Choose model: "))
    return models[index]


def load_trained_model(model_path, method, feature_type):
    if method == "SNN" and feature_type == "wavelet":
        model = load_model(
            model_path,
            custom_objects={
                "euclidan_distance": functions.euclidan_distance,
                "euclidan_dist_output_shape": functions.euclidan_dist_output_shape,
            },
        )
    elif method == "SNN" and feature_type == "six_fold":
        model = load_model(
            model_path,
            custom_objects={
                "euclidan_distance": functions.euclidan_distance,
                "euclidan_dist_output_shape": functions.euclidan_dist_output_shape,
            },
        )
    else:
        model = load_model(model_path)
    return model


def get_feature_and_method(model_path, method=None):
    if "CNN" in model_path:
        method = "CNN"
    elif "SNN" in model_path:
        method = "SNN"
    else:
        if method is None:
            is_cnn = int(input("SNN (0) or CNN (1): "))
            if is_cnn == "1":
                method = "CNN"
            elif is_cnn == "0":
                method = "SNN"
        else:
            method = method
    for feature_type in FEATURES:
        if feature_type == "local":
            feature_type = "local."
        if feature_type in model_path:
            if feature_type == "local.":
                feature_type = "local"
            feature = feature_type
            break
        else:
            feature = "NotFound"
    if feature == "NotFound":
        is_feature = int(input("Without features (0) or with features (1): "))
        if is_feature == 1:
            print("Available features: ")
            for i in range(len(FEATURES)):
                print(f"{i}: {FEATURES[i]}")
            index = int(input("Choose feature: "))
            feature = FEATURES[index]
        else:
            feature = "None"
    return feature, method


def evaluate_model(
    model_path,
    dataset,
    method,
    feature_type,
    data_dir="test",
    image_width=150,
    image_height=150,
    augmented=False,
    size=0,
):
    model = load_trained_model(model_path, method, feature_type)
    if method == "CNN":
        data, labels = loader.loader_for_cnn(
            data_dir,
            image_width=image_width,
            image_height=image_height,
            dataset=dataset,
            augmented=augmented,
            size=size,
        )
        data, labels = functions.shuffle_data(data, labels)
        if feature_type != "None" and feature_type != "local_solo":
            feature = functions.add_features(
                data, is_pair=False, feature_type=feature_type
            )
            result = model.evaluate(
                x=([data[:,], feature[:,]]),
                y=labels,
            )
        elif feature_type == "local_solo":
            feature = functions.add_features(
                data, is_pair=False, feature_type=feature_type
            )
            result = model.evaluate(x=feature, y=labels)
        else:
            result = model.evaluate(x=data, y=labels)
    elif method == "SNN":
        pair, labels = loader.loader_for_snn(
            data_dir,
            image_width=image_width,
            image_height=image_height,
            dataset=dataset,
            augmented=augmented,
            size=size,
        )
        pair, labels = functions.shuffle_data(pair, labels)
        if feature_type != "None" and feature_type != "local_solo":
            feature = functions.add_features(
                pair, is_pair=True, feature_type=feature_type
            )
            result = model.evaluate(
                x=(
                    [
                        pair[:, 0, :, :],
                        feature[:, 0],
                        pair[:, 1, :, :],
                        feature[:, 1],
                    ]
                ),
                y=labels,
            )
        elif feature_type == "local_solo":
            feature = functions.add_features(
                pair, is_pair=True, feature_type=feature_type
            )
            result = model.evaluate(
                x=([feature[:, 0], feature[:, 1]]),
                y=labels,
            )
        else:
            result = model.evaluate(x=([pair[:, 0, :, :], pair[:, 1, :, :]]), y=labels)

    return result


def model_evaluation(
    data_dir="test",
    dataset="czech_test",
    model_dir="models/server/czech",
    model_path=None,
):
    if model_path is None:
        model_path = get_model(model_dir)

    feature_type, method = get_feature_and_method(model_path)

    width = int(input("Image width: "))
    height = int(input("Image height: "))
    augment_input = input("Augment image (y/n): ")
    if augment_input == "n" or augment_input == "N":
        augmented = False
    else:
        augmented = True
    size = int(input("Enter size (0 or number): "))

    result = evaluate_model(
        model_path,
        dataset=dataset,
        method=method,
        feature_type=feature_type,
        data_dir=data_dir,
        image_height=height,
        image_width=width,
        augmented=augmented,
        size=size,
    )
    print(f"For model {model_path} loss = {result[0]:.3f} and acc = {result[1]:.3f}")
    return


def predict_images(model_path, image_array, method, feature_type):
    model = load_trained_model(model_path, method, feature_type)
    if feature_type != "None":
        if method == "CNN":
            feature = functions.add_features(
                image_array, is_pair=False, feature_type=feature_type
            )
        else:
            feature = functions.add_features(
                image_array, is_pair=True, feature_type=feature_type
            )

    if method == "CNN":
        if feature_type != "None" and feature_type != "local_solo":
            result = model.predict(x=([image_array[:,], feature[:,]]))
        elif feature_type == "local_solo":
            result = model.evaluate(x=feature)
            result = result[:1]
        else:
            result = model.evaluate(x=image_array)
    elif method == "SNN":
        if feature_type != "None" and feature_type != "local_solo":
            result = model.evaluate(
                x=(
                    [
                        image_array[:, 0, :, :],
                        feature[:, 0],
                        image_array[:, 1, :, :],
                        feature[:, 1],
                    ]
                )
            )
        elif feature_type == "local_solo":
            result = model.evaluate(x=([feature[:, 0], feature[:, 1]]))
            result = result[:1]
        else:
            result = model.evaluate(
                x=([image_array[:, 0, :, :], image_array[:, 1, :, :]])
            )
    return result


def model_prediction(
    img_path_array, method=None, model_dir="models/server/czech", model_path=None
):
    ans = -1
    if model_path is None:
        model_path = get_model(model_dir)
    width = int(input("Image width: "))
    height = int(input("Image height: "))
    feature_type, method = get_feature_and_method(model_path, method=method)
    if method == "CNN" and feature_type == "None":
        ans = int(input("Interpret prediction? No(0), with SHAP(1), with Grad-CAM(2)"))
    if ans == 1 or ans == 2:
        save_path = input("Save interpretation? No/save_path: ")
        if (
            save_path == "No"
            or save_path == "no"
            or save_path == "n"
            or save_path == "N"
        ):
            save_path = None
    image_array = []
    for image_path in img_path_array:
        if method == "CNN":
            image = loader.convert_to_image(image_path, img_w=width, img_h=height)
            image_array.append(image)
        else:
            pair1 = loader.convert_to_image(image_path[0], img_w=width, img_h=height)
            pair2 = loader.convert_to_image(image_path[1], img_w=width, img_h=height)
            image_array.append([pair1, pair2])
    image_array = np.array(image_array, dtype=np.float32)

    predictions = predict_images(model_path, image_array, method, feature_type)
    labels = functions.prediction_to_label(predictions)
    for i, prediction in enumerate(predictions):
        print(f"For image {i+1} predicted {labels[i]} with probability {prediction}")
    if ans == 1:
        if len(image_array) <= 1:
            shap_visualization(
                save_path=save_path,
                model_path=model_path,
                data=image_array,
                prediction=predictions,
            )
        else:
            for i, prediction in enumerate(predictions):
                if save_path is not None:
                    save_path = input(f"Enter path for image {i + 1}: ")
                shap_visualization(
                    save_path=save_path,
                    model_path=model_path,
                    data=image_array[i],
                    prediction=prediction,
                )
    if ans == 2:
        if len(image_array) <= 1:
            gradcam_visualization(
                save_path=save_path,
                model_path=model_path,
                data=image_array[0],
                prediction=predictions,
            )
        else:
            for i, prediction in enumerate(predictions):
                if save_path is not None:
                    save_path = input(f"Enter path for image {i}: ")
                    gradcam_visualization(
                        save_path=save_path,
                        model_path=model_path,
                        data=image_array[i],
                        prediction=prediction,
                    )
    return


def load_test_and_predict(model, data_dir="test", dataset="czech_test"):
    data, labels = loader.loader_for_cnn(data_dir=data_dir, dataset=dataset)
    data, labels = functions.shuffle_data(data, labels)
    nu_preds = 1
    data = data[:nu_preds]
    prediction = model.predict(data)
    print(labels)
    for i in range(len(prediction)):
        print(f"predictions: {prediction[i]} for label: {labels[i]}")
    return data, prediction


def shap_visualization(
    img_path=None,
    save_path=None,
    model_path="models/server/czech/CNN_cz_train_czech_None.h5",
    data=None,
    prediction=None,
):
    model = load_model(model_path)
    if img_path is None and data is None and prediction is None:
        data, prediction = load_test_and_predict(model)
    elif img_path is not None:
        data = loader.convert_to_image(img_path)
        data = np.expand_dims(data, axis=0)
        data = np.array(data, dtype=np.float32)
        prediction = model.predict(data)
    else:
        data = np.expand_dims(data, axis=0)
        data = np.array(data, dtype=np.float32)

    pred_label = prediction_to_label(prediction)
    visualize_with_shap(data=data, model=model, pred=pred_label, save_path=save_path)


def gradcam_visualization(
    img_path=None,
    save_path=None,
    model_path="models/server/czech/CNN_cz_train_czech_None.h5",
    data=None,
    prediction=None,
):
    alpha = 0.4
    model = load_model(model_path)
    for layer in model.layers:
        if "conv" in layer.name:
            last_layer = layer.name
    if img_path is None and data is None and prediction is None:
        data, prediction = load_test_and_predict(model)
        image = data[0]
    elif img_path is not None:
        data = loader.convert_to_image(img_path)
        data = np.expand_dims(data, axis=0)
        data = np.array(data, dtype=np.float32)
        prediction = model.predict(data)
        image = data[0]
        print(f"predictions: {prediction} for img: {img_path}")
    else:
        data = np.expand_dims(data, axis=0)
        data = np.array(data, dtype=np.float32)
        image = data[0]

    pred_label = prediction_to_label(prediction)

    heatmap = make_gradcam_heatmap(data, model, last_layer, pred_index=None)

    overlaid_image = overlay_heatmap(image, heatmap, alpha=alpha)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(pred_label[0][0])
    axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Heatmap")
    axes[1].axis("off")
    axes[2].imshow(cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Overlaid Image")
    axes[2].axis("off")
    if save_path:
        plt.savefig(save_path)
        print("fig saved")
    else:
        plt.show(block=True)


def validator_for_package(
    datadir="models/server/package", feature_type="None", method="CNN"
):
    if feature_type == "local":
        feature_type = "local."
    models = glob.glob(datadir + "/*.h5")
    models_for_validation = []
    results = []
    for model_path in models:
        if method in model_path and feature_type in model_path:
            models_for_validation.append(model_path)
    if feature_type == "local.":
        feature_type = "local"

    for model_path in models_for_validation:
        print(f"___________________________{model_path}_____________________________")
        if "cedar" in model_path:
            result = evaluate_model(model_path, "cedar_test", method, feature_type)
        if "dutch" in model_path:
            result = evaluate_model(model_path, "dutch_test", method, feature_type)
        if "gdps" in model_path:
            result = evaluate_model(model_path, "gdps_test", method, feature_type)
        if "all" in model_path:
            result = evaluate_model(model_path, "all_test", method, feature_type)
        if "czech" in model_path:
            result = evaluate_model(model_path, "czech_test", method, feature_type)

        print(
            f"For model {model_path} loss = {result[0]:.3f} and acc = {result[1]:.3f}"
        )
        results.append([model_path, result])

    return results


def validate_czech_on_best_package(datadir="models/server/package"):
    CNN_models = [
        "CNN_cedar_strokes.h5",
        "CNN_dutch_histogram.h5",
        "CNN_gdps_None.h5",
        "CNN_all_None.h5",
    ]
    SNN_models = [
        "SNN_cedar_local.h5",
        "SNN_dutch_None.h5",
        "SNN_gdps_tri_surface.h5",
        "SNN_all_tri_shape.h5",
    ]
    results = []
    for model in CNN_models:
        model_path = datadir + "/" + model
        if "strokes" in model:
            feature_type = "strokes"
        elif "histogram" in model:
            feature_type = "histogram"
        else:
            feature_type = "None"
        result = evaluate_model(model_path, "czech_test", "CNN", feature_type)
        print(f"For model {model} loss = {result[0]:.3f} and acc = {result[1]:.3f}")
        results.append([model, result])
    for model in SNN_models:
        model_path = datadir + "/" + model
        if "local" in model:
            feature_type = "local"
        elif "tri_surface" in model:
            feature_type = "tri_surface"
        elif "tri_shape" in model:
            feature_type = "tri_shape"
        else:
            feature_type = "None"
        result = evaluate_model(model_path, "czech_test", "SNN", feature_type)
        print(f"For model {model} loss = {result[0]:.3f} and acc = {result[1]:.3f}")
        results.append([model, result])
    return results
