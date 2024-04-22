import glob
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras
import loader
from functions import visualize_with_shap


def test_model(datadir= "fromServer"):
    models = glob.glob(datadir + "/*.h5")
    for i in range(len(models)):
        print(f'{i}: {models[i]}')
    index = int(input("Choose model: "))
    model_path = models[index]
    is_cnn = int(input("SNN (0) or CNN (1): "))
    num_test_samples = int(input("Number of test samples: "))
    batch_size = int(input("Batch size: "))
    width = int(input("Image width: "))
    height = int(input("Image height: "))

    model = load_model(model_path)

    if is_cnn == 1:
        num_test_samples = int(num_test_samples / 2)
        data, labels = loader.loader_for_cnn(
            data_dir='test',
            image_width=width,
            image_height=height,
            augmented=False,
            size=num_test_samples,
            dataset='test',
        )
    else:
        print(num_test_samples)
        pairs, labels = loader.loader_for_snn(
            data_dir='test_cedar',
            image_width=width,
            image_height=height,
            size=num_test_samples,
            dataset='test',
        )
    is_eval = 200
    while is_eval != -1:
        is_eval = int(input("Evaluate(0) or predict(1) or end(-1): "))
        if is_eval == 0:
            if is_cnn == 0:
                result = model.evaluate(
                    x=([pairs[:, 0, :,:], pairs[:,1,:,:]]),
                    y=labels,
                    batch_size=batch_size,
                )
            else:
                result = model.evaluate(x=data, y=labels, batch_size=batch_size)
            print(f'test loss and acc = {result}')
        elif is_eval == 1:
            num_of_pred = int(input("Number of predictions: "))
            if is_cnn == 0:
                new_pairs = pairs[:num_of_pred]
                prediction = model.predict([new_pairs[:, 0, :,:], new_pairs[:,1,:,:]])
            else:
                new_data = data[:num_of_pred]
                prediction = model.predict(new_data)
            print(f'prediction shape: {prediction.shape}')
            for i in range(len(prediction)):
                print(f'predictions: {prediction[i]} for lable: {labels[i]}')







