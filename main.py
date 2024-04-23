from __future__ import  absolute_import, division, print_function, unicode_literals
# IMPORTING ALLES :))
import datetime
import sys
import numpy as np
import pickle
import os
import matplotlib

import functions

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import tensorflow as tf


import cv2
import time
import itertools
import random

from sklearn.utils import shuffle


import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
import glob
import logging


import loader
import model

#IMG_HEIGHT = 150
#IMG_WIDTH = 150
CHANNELS = 1
#LABELS = np.array(["Forged", "Genuine"])
#BATCH_SIZE = 32
#EPOCH_SIZE = 10
DATASETS = ["cedar", "chinese", "dutch", "hindi", "bengali", "GDPS", "all"]
FEATURES = ["None", "strokes", "local", "local_solo", "wavelet", "tri_shape", "tri_surface", "sixfold"]
                      
image_shape = (None, 100, 100, 3)

def cnn_train(epochs = 100, batch_size = 32, img_width = 150, img_height = 150, dataset='cedar', augmented=False, save_name=None, size=None, feature_type=None):
    if save_name is None:
        csv_name = "CNN"
    else:
        csv_name = save_name

    data, labels = loader.loader_for_cnn(image_width=img_width, image_height=img_height, dataset=dataset,
                                         augmented=augmented, size=size)
    if feature_type is not None:
        feature = functions.add_features(data, is_pair=False, feature_type=feature_type)
        if feature_type == "wavelet":
            CNNMODEL = model.cnn_feature_model(image_shape=(img_width,img_height, CHANNELS), feature_type=feature_type, feature_shape=feature.shape[1])
        elif feature_type == "local" or feature_type == "local_solo":
            CNNMODEL = model.cnn_feature_model(image_shape=(img_width,img_height, CHANNELS),feature_type=feature_type,
                                               feature_shape=(feature.shape[1], feature.shape[2], feature.shape[3], feature.shape[4]))
        else:
            CNNMODEL = model.cnn_feature_model(image_shape=(img_width, img_height, CHANNELS), feature_type=feature_type)
    else:
        CNNMODEL = model.cnn_model(image_shape=(img_width, img_height, CHANNELS))


    if feature_type is not None and feature_type != "local_solo":
        hist = CNNMODEL.fit(x=([data[:,], feature[:,]]),
                            y=labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2,
                            shuffle=True,
                            callbacks=functions.callbacks_schelude_lr((csv_name + ".csv"))
                            )
    elif feature_type == "local_solo":
        hist = CNNMODEL.fit(x=feature,
                            y=labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2,
                            shuffle=True,
                            callbacks=functions.callbacks_schelude_lr((csv_name + ".csv"))
                            )
    else:
        hist = CNNMODEL.fit(x=data,
                            y=labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2,
                            shuffle=True,
                            callbacks=functions.callbacks_schelude_lr((csv_name + ".csv"))
                            )
    # MODEL.build(image_shape)
    CNNMODEL.summary()
    #
    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(hist.history['loss'], color='teal', label='loss')
    # plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    # fig.suptitle('Loss', fontsize=20)
    # plt.legend(loc="upper left")
    # plt.show()
    #
    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    # plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    # fig.suptitle('Accuracy', fontsize=20)
    # plt.legend(loc="upper left")
    # plt.show()
    if save_name is not None:
        print("Model saved")
        CNNMODEL.save(os.path.join('models', (save_name+'.h5')))

    print('\n\n\n\nAAAAAALLL DONE')

def snn_train(epochs = 100, batch_size = 32, img_width = 150, img_height = 150, dataset='cedar', size=2000, feature_type=None, gdps_size=None, save_name=None):
    if save_name is None:
        csv_name = "SNN"
    else:
        csv_name = save_name
    data_pairs, data_labels = loader.loader_for_snn(image_width=img_width, image_height=img_height,dataset=dataset, size=size, gdps_size=gdps_size)
    if feature_type is not None:
        feature = functions.add_features(data_pairs, feature_type=feature_type)
        if feature_type == "wavelet":
            SNNMODEL = model.snn_model(image_shape=(img_width, img_height, CHANNELS), feature_shape=feature.shape[2], feature_type=feature_type)
        elif feature_type == "local" or feature_type == "local_solo":
            SNNMODEL = model.snn_model(image_shape=(img_width, img_height, CHANNELS), feature_type=feature_type,
                                       feature_shape=(feature.shape[2], feature.shape[3], feature.shape[4], feature.shape[5]))
        else:
            SNNMODEL = model.snn_model(image_shape=(img_width, img_height, CHANNELS), feature_type=feature_type)
    else:
        SNNMODEL = model.snn_model(image_shape=(img_width, img_height, CHANNELS))

    if feature_type is not None and feature_type != "local_solo":
        hist = SNNMODEL.fit(
            x=([data_pairs[:, 0, :,:],feature[:, 0], data_pairs[:,1,:,:], feature[:,1]]),
            y=data_labels,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_split=0.2,
            callbacks=functions.callbacks_schelude_lr((csv_name + ".csv"))
        )
    elif feature_type == "local_solo":
        hist = SNNMODEL.fit(
            x=([feature[:, 0], feature[:,1]]),
            y=data_labels,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_split=0.2,
            callbacks=functions.callbacks_schelude_lr((csv_name + ".csv"))
        )
    else:
        hist = SNNMODEL.fit(
            x=([data_pairs[:, 0, :,:], data_pairs[:,1,:,:]]),
            y=data_labels,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_split=0.2,
            callbacks=functions.callbacks_schelude_lr((csv_name + ".csv"))
        )

    if save_name is not None:
        print("Model saved")
        SNNMODEL.save(os.path.join('models', (save_name + '.h5')))
    # SNNMODEL.summary()

    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(hist.history['loss'], color='teal', label='loss')
    # plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    # fig.suptitle('Loss', fontsize=20)
    # plt.legend(loc="upper left")
    # plt.show()
    #
    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    # plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    # fig.suptitle('Accuracy', fontsize=20)
    # plt.legend(loc="upper left")
    # plt.show(block=True)

def continue_on_snn(model, epochs = 100, batch_size = 32, img_width = 150, img_height = 150, dataset='cedar', size=5000):
    loaded_model = load_model(model)
    data_pairs, data_labels = loader.loader_for_snn(image_width=img_width, image_height=img_height, size=size)
    hist = loaded_model.fit(
        x=([data_pairs[:, 0, :,:], data_pairs[:,1,:,:]]),
        y=data_labels,
        #steps_per_epoch= int(len(train_pairs)/BATCH_SIZE),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_split=0.2,
        callbacks=functions.callbacks_schelude_lr('continueOnSNN.csv')
    )

    loaded_model.save(os.path.join('models', 'SnnSignatureVerificatorLoaded.h5'))

    loaded_model.summary()

    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(hist.history['loss'], color='teal', label='loss')
    # plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    # fig.suptitle('Loss', fontsize=20)
    # plt.legend(loc="upper left")
    # plt.show()
    #
    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    # plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    # fig.suptitle('Accuracy', fontsize=20)
    # plt.legend(loc="upper left")
    # plt.show()

    print('\n\n\n\nAAAAAALLL DONE')


def main():
    epochsize = int(input("Number of epochs: "))
    batchsize = int(input("Batch size: "))
    width = 150
    height = 150
    save = input("Name of model to save: ")
    if save == "":
        save = None
    for i in range(len(DATASETS)):
        print(f"{i}: {DATASETS[i]}")
    dataset = int(input("Chose dataset: "))
    dataset = DATASETS[dataset]
    for i in range(len(FEATURES)):
        print(f"{i}: {FEATURES[i]}")
    feature = int(input("Chose feature: "))
    if feature == 0:
        feature_type = None
    else:
        feature_type = FEATURES[feature]

    ans = int(input('Do you wanna activate CNN(0) or SNN(1):  '))
    if ans == 0:
        augmented = bool(input("Augmented data (True, ENTER): "))
        size = int(input("Size of data: "))
        cnn_train(epochs=epochsize, batch_size=batchsize, img_width=width, img_height=height, augmented=augmented, dataset=dataset, size=size, save_name=save)
    elif ans == 1:
        size = int(input("Size of pairs: "))

        snn_train(epochs=epochsize, batch_size=batchsize, img_width=width, img_height=height, size=size, dataset=dataset, save_name=save, feature_type=feature_type)
    else:
        print("nothings gonna happen :)")
        return



if __name__ == '__main__':
    main()
