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
                      
image_shape = (None, 100, 100, 3)

def cnn_train(epochs = 100, batch_size = 32, img_width = 150, img_height = 150, dataset='cedar'):
    train_ds, val_ds = loader.tensor_loader_for_cnn(batch_size=batch_size, image_width=img_width,
                                                    image_height=img_height)

    CNNMODEL = model.cnn_model()
    hist = CNNMODEL.fit(train_ds, batch_size=batch_size, epochs=epochs,
                        # steps_per_epoch=len(train_ds)/BATCH_SIZE,
                        validation_data=val_ds,
                        shuffle=True,
                        callbacks=functions.callbacks_schelude_lr('TSCNN.csv')
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
    #
    # pre = Precision()
    # re = Recall()
    # acc = BinaryAccuracy()

    CNNMODEL.save(os.path.join('models', 'cnnSignatureVerificator.h5'))

    print('\n\n\n\nAAAAAALLL DONE')

def cnn_train_augmented(epochs = 100, batch_size = 32, img_width = 150, img_height = 150, dataset='cedar', augmented=False):
    data, labels = loader.loader_for_cnn(image_width=img_width, image_height=img_height, dataset=dataset, augmented=augmented)
    CNNMODEL = model.cnn_model(image_shape=(img_width,img_height, CHANNELS))
    hist = CNNMODEL.fit(x=data,
                        y=labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        # steps_per_epoch=len(train_ds)/BATCH_SIZE,
                        validation_split=0.2,
                        shuffle=True,
                        callbacks=functions.callbacks_schelude_lr('CNN.csv')
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

    CNNMODEL.save(os.path.join('models', 'cnnSignatureVerificatorAugmented.h5'))

    print('\n\n\n\nAAAAAALLL DONE')

def snn_train(epochs = 100, batch_size = 32, img_width = 150, img_height = 150, dataset='cedar', size=2000, type=None):

    data_pairs, data_labels = loader.loader_for_snn(image_width=img_width, image_height=img_height, size=size)
    if type is not None:
        feature = functions.add_features(data_pairs, type=type)
        if type == "wavelet":
            SNNMODEL = model.snn_model(image_shape=(img_width, img_height, CHANNELS), feature_shape=feature.shape[2], feature_type=type)
        else:
            SNNMODEL = model.snn_model(image_shape=(img_width, img_height, CHANNELS), feature_type=type)
    else:
        SNNMODEL = model.snn_model(image_shape=(img_width, img_height, CHANNELS))


    hist = SNNMODEL.fit(
        x=([data_pairs[:, 0, :,:],feature[:, 0], data_pairs[:,1,:,:], feature[:,1]]),
        y=data_labels,
        #steps_per_epoch= int(len(train_pairs)/BATCH_SIZE),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_split=0.2,
        callbacks=functions.callbacks_schelude_lr('SNN.csv')
        #validation_data=([val_pairs[:,0,:,:], val_pairs[:,1,:,:]], val_labels),
        #validation_steps=int(len(val_pairs)/BATCH_SIZE),
        #callbacks=functions.callbacks()
    )

    # SNNMODEL.save(os.path.join('models', 'SnnSignatureVerificatorFinal.h5'))
    SNNMODEL.summary()

    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(hist.history['loss'], color='teal', label='loss')
    # plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    # fig.suptitle('Loss', fontsize=20)
    # plt.legend(loc="upper left")
    # plt.show()
    #
    fig = plt.figure(figsize=(7, 7))
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show(block=True)

def continue_on_cnn():
    pass

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


def main():
    epochsize = int(input("Number of epochs: "))
    batchsize = int(input("Batch size: "))
    width = int(input("Image width: "))
    height = int(input("Image height: "))

    ans = int(input('Do you wanna activate CNN(0) CNN AUGMENTEd(1) or SNN(2) continue training CNN(3) or SNN(4):  '))
    if ans == 0:cnn_train(epochs=epochsize, batch_size=batchsize, img_width=width, img_height=height)
    elif ans == 1:
        augmented = bool(input("Augmented data (True, ENTER): "))
        cnn_train_augmented(epochs=epochsize, batch_size=batchsize, img_width=width, img_height=height, augmented=augmented)
    elif ans == 2:
        size = int(input("Size of pairs: "))
        snn_train(epochs=epochsize, batch_size=batchsize, img_width=width, img_height=height, size=size)
    # elif ans == 3: continue_on_cnn()
    # elif ans == 4: continue_on_snn('models/SnnSignatureVerificatorFinal.h5')
    else:
        print("nothings gonna happen :)")
        return



if __name__ == '__main__':
    main()
