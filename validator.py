import glob

import cv2
import numpy as np
from main import FEATURES

from tensorflow.keras.models import load_model
import functions
import loader
from functions import visualize_with_shap, prediction_to_label, overlay_heatmap
from model import make_gradcam_heatmap
from matplotlib import pyplot as plt

def evaluate_model(datadir="fromServer", model_path=None, dataset="czech_test", data_dir="test", ):
    if model_path is None:
        models = glob.glob(datadir + "/*.h5")
        print("Available models: ")
        for i in range(len(models)):
            print(f"{i}: {models[i]}")
        index = int(input("Choose model: "))
        model_path = models[index]

    is_cnn = int(input("SNN (0) or CNN (1): "))
    is_feature = int(input("Without features (0) or with features (1): "))
    if is_feature == 1:
        print("Available features: ")
        for i in range(len(FEATURES)):
            print(f"{i}: {FEATURES[i]}")
        index = int(input("Choose feature: "))
        feature_type = FEATURES[i]
    else:
        feature_type = None

    batch_size = int(input("Batch size: "))
    width = int(input("Image width: "))
    height = int(input("Image height: "))
    augment_input = int(input("Augment image (y/n): "))
    if augment_input == "n" or augment_input == "N":
        augmented = False
    else:
        augmented = True

    model = load_model(model_path)

    if is_cnn == 1:
        data, labels = loader.loader_for_cnn(
            data_dir=data_dir,
            image_width=width,
            image_height=height,
            augmented=augmented,
            dataset=dataset,
        )
    else:
        pairs, labels = loader.loader_for_snn(
            data_dir=data_dir,
            image_width=width,
            image_height=height,
            dataset="cedar_test",
        )
    is_eval = 200
    while is_eval != -1:
        is_eval = int(input("Evaluate(0) or predict(1) or end(-1): "))
        if is_eval == 0:
            if is_cnn == 0:
                result = model.evaluate(
                    x=([pairs[:, 0, :, :], pairs[:, 1, :, :]]),
                    y=labels,
                    batch_size=batch_size,
                )
            else:
                result = model.evaluate(x=data, y=labels, batch_size=batch_size)
            print(f"test loss and acc = {result}")
        elif is_eval == 1:
            num_of_pred = int(input("Number of predictions: "))
            if is_cnn == 0:
                new_pairs = pairs[:num_of_pred]
                prediction = model.predict(
                    [new_pairs[:, 0, :, :], new_pairs[:, 1, :, :]]
                )
            else:
                new_data = data[:num_of_pred]
                prediction = model.predict(new_data)
            print(f"prediction shape: {prediction.shape}")
            for i in range(len(prediction)):
                print(f"predictions: {prediction[i]} for label: {labels[i]}")


#TODO dodelat pro ostatni datasety a uzivatelsky
def shap_visualization(save_img=None):
    model = load_model("models/server/czech/CNN_cz_train_czech_None.h5")
    data, labels = loader.loader_for_cnn("test", dataset="czech_test")
    data, labels = functions.shuffle_data(data, labels)
    nu_preds = 1
    data = data[:nu_preds]
    prediction = model.predict(data)
    print(labels)
    for i in range(len(prediction)):
        print(f"predictions: {prediction[i]} for label: {labels[i]}")
    pred_label = prediction_to_label(prediction)
    visualize_with_shap(data=data, model=model, pred=pred_label, save_path=save_img)

def gradcam_visualization(img_path=None, save_path=None, model_path="models/server/czech/CNN_cz_train_czech_None.h5"):
    alpha = 0.4
    model = load_model(model_path)
    for layer in model.layers:
        if "conv" in layer.name:
            last_layer = layer.name
    if img_path is None:
        data, labels = loader.loader_for_cnn("test", dataset="czech_test")
        data, labels = functions.shuffle_data(data, labels)
        nu_preds = 1
        data = data[:nu_preds]
        prediction = model.predict(data)
        print(labels)
        for i in range(len(prediction)):
            print(f"predictions: {prediction[i]} for label: {labels[i]}")
    else:
        data = loader.convert_to_image(img_path)
        data = np.expand_dims(data, axis=0)
        data = np.array(data, dtype=np.float32)
        prediction = model.predict(data)
        print(print(f"predictions: {prediction} for img: {img_path}"))

    heatmap = make_gradcam_heatmap(data, model, last_layer, pred_index=None)

    image = data[0]

    overlaid_image = overlay_heatmap(image, heatmap, alpha=alpha)

    # Plot all three images side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title('Původní obrázek')
    axes[0].axis('off')

    # Plot the heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Heatmapa')
    axes[1].axis('off')

    # Plot the overlaid image
    axes[2].imshow(cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Aplikovaná heatmapa')
    axes[2].axis('off')

    if save_path:
        plt.savefig(save_path)
        print("fig saved")
    else:
        plt.show(block=True)

def evaluate_model(model_path, dataset, method, feature_type):
    if method == "SNN" and  feature_type == "wavelet":
        model = load_model(model_path, custom_objects={
            "euclidan_distance": functions.euclidan_distance,
            "euclidan_dist_output_shape": functions.euclidan_dist_output_shape,
        })
    elif method == "SNN" and  feature_type == "six_fold":
        model = load_model(model_path, custom_objects={
            "euclidan_distance": functions.euclidan_distance,
            "euclidan_dist_output_shape": functions.euclidan_dist_output_shape,
        })
    else:
        model = load_model(model_path)

    if method == "CNN":
        data, labels = loader.loader_for_cnn("test", dataset=dataset)
        data, labels = functions.shuffle_data(data, labels)
        if feature_type != "None" and feature_type != "local_solo":
            feature = functions.add_features(data, is_pair=False, feature_type=feature_type)
            result = model.evaluate(x=([data[:,], feature[:,]]),y=labels,)
        elif feature_type == "local_solo":
            feature = functions.add_features(data, is_pair=False, feature_type=feature_type)
            result = model.evaluate(x=feature, y=labels)
        else:
            result = model.evaluate(x=data, y=labels)
    elif method == "SNN":
        pair, labels = loader.loader_for_snn("test", dataset=dataset)
        pair, labels = functions.shuffle_data(pair, labels)
        if feature_type != "None" and feature_type != "local_solo":
            feature = functions.add_features(pair, is_pair=True, feature_type=feature_type)
            result = model.evaluate(x=([
                    pair[:, 0, :, :],
                    feature[:, 0],
                    pair[:, 1, :, :],
                    feature[:, 1],
                ]), y=labels)
        elif feature_type == "local_solo":
            feature = functions.add_features(pair, is_pair=True, feature_type=feature_type)
            result = model.evaluate(x=([feature[:, 0], feature[:, 1]]), y=labels,)
        else:
            result = model.evaluate(x=([pair[:,0,:,:], pair[:,1,:,:]]), y=labels)

    return result




def validator_for_package(datadir="models/server/package", feature_type="None", method="CNN"):
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
            result =evaluate_model(model_path, "all_test", method, feature_type)
        if "czech" in model_path:
            result = evaluate_model(model_path, "czech_test", method, feature_type)

        print(f"For model {model_path} loss = {result[0]:.3f} and acc = {result[1]:.3f}")
        results.append([model_path, result])

    return results

def validate_czech_on_best_package(datadir="models/server/package"):
    CNN_models = ["CNN_cedar_strokes.h5","CNN_dutch_histogram.h5", "CNN_gdps_None.h5", "CNN_all_None.h5"]
    SNN_models = ["SNN_cedar_local.h5", "SNN_dutch_None.h5", "SNN_gdps_tri_surface.h5", "SNN_all_tri_shape.h5"]
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