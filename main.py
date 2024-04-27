from __future__ import  absolute_import, division, print_function, unicode_literals
import os
import matplotlib

import functions

matplotlib.use('Agg')


import tensorflow as tf
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
from tensorflow.keras.models import load_model


import loader
import model

#IMG_HEIGHT = 150
#IMG_WIDTH = 150
CHANNELS = 1
#LABELS = np.array(["Forged", "Genuine"])
#BATCH_SIZE = 32
#EPOCH_SIZE = 10
DATASETS = ["cedar", "chinese", "dutch", "hindi", "bengali", "gdps", "all"]
FEATURES = ["None", "strokes", "histogram" "local", "local_solo", "wavelet", "tri_shape", "tri_surface", "six_fold"]
                      
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
        if feature_type == "wavelet" or feature_type == "histogram":
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
    #CNNMODEL.summary()
    #
    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(hist.history['loss'], color='teal', label='loss')
    # plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    # fig.suptitle('Loss', fontsize=20)
    # plt.legend(loc="upper left")
    # plt.show(block=True)
    #
    # fig = plt.figure(figsize=(7, 7))
    # plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    # plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    # fig.suptitle('Accuracy', fontsize=20)
    # plt.legend(loc="upper left")
    # plt.show(block=True)

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
        if feature_type == "wavelet" or feature_type == "histogram":
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
    # plt.show(block=True)
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

    # loaded_model.summary()

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
    augmented = False


    ans = int(input('Do you wanna activate CNN(0) or SNN(1) or Package CNN(2) or  Package SNN(3):  '))
    if ans == 0 or ans == 1:
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


    if ans == 0:
        augmented = bool(input("Augmented data (True, ENTER): "))
        size = int(input("Size of data: "))
        cnn_train(epochs=epochsize, batch_size=batchsize, img_width=width, img_height=height, augmented=augmented, dataset=dataset, size=size, save_name=save, feature_type=feature_type)
    elif ans == 1:
        size = int(input("Size of pairs: "))

        snn_train(epochs=epochsize, batch_size=batchsize, img_width=width, img_height=height, size=size, dataset=dataset, save_name=save, feature_type=feature_type)

    elif ans == 2 or ans == 3 :
        model_dir = input("Enter output dir: ")
        if ans == 2:
            size = int(input("Size of data: "))
            augmented = bool(input("Augmented data (True, ENTER): "))
        else:
            size = int(input("Size of pairs: "))
        for dataset in DATASETS:
            for feature in FEATURES:
                if feature == "None":
                    feature = None
                if ans == 2:
                    save = model_dir + "/" + "CNN_" + dataset + "_" + str(feature)
                    print(f"Training for CNN with {dataset} and {feature}")
                    cnn_train(epochs=epochsize, batch_size=batchsize, img_width=width, img_height=height, augmented=augmented, dataset=dataset, size=size, save_name=save, feature_type=feature)
                else:
                    save = model_dir + "/" + "SNN_" + dataset + "_" + str(feature)
                    print(f"Training for SNN with {dataset} and {feature}")
                    snn_train(epochs=epochsize, batch_size=batchsize, img_width=width, img_height=height, size=size, dataset=dataset, save_name=save, feature_type=feature)
    else:
        print("nothings gonna happen :)")
        return



if __name__ == '__main__':
    main()
