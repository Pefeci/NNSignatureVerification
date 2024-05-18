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

def shap_visualization():
    model = load_model("models/CNN_cedar_None.h5")
    data, labels = loader.loader_for_cnn("test", image_width=100, image_height=100, dataset="cedar_test")
    data, labels = functions.shuffle_data(data, labels)
    nu_preds = 1
    data = data[:nu_preds]
    prediction = model.predict(data)
    print(labels)
    for i in range(len(prediction)):
        print(f"predictions: {prediction[i]} for label: {labels[i]}")
    pred_label = prediction_to_label(prediction)
    visualize_with_shap(data=data, model=model, pred=pred_label)

def gradcam_visualization():
    alpha = 0.4
    model = load_model("models/CNN_cedar_None.h5")
    for layer in model.layers:
        if "conv" in layer.name:
            last_layer = layer.name
    data, labels = loader.loader_for_cnn("test", image_width=100, image_height=100, dataset="cedar_test")
    data, labels = functions.shuffle_data(data, labels)
    nu_preds = 1
    data = data[:nu_preds]
    prediction = model.predict(data)
    print(labels)
    for i in range(len(prediction)):
        print(f"predictions: {prediction[i]} for label: {labels[i]}")


    heatmap = make_gradcam_heatmap(data, model, last_layer, pred_index=None)

    image = data[0]
    overlaid_image = overlay_heatmap(image, heatmap)

    # Plot all three images side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot the heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Heatmap')
    axes[1].axis('off')

    # Plot the overlaid image
    axes[2].imshow(cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Overlay')
    axes[2].axis('off')


    plt.savefig('C:/Users/Pefeci/Desktop/plotGrad.png')
    print("fig saved")