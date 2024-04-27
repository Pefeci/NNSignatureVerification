"""
@author: Petr Čírtek
"""

from keras import models
import tensorflow as tf
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D, \
    BatchNormalization, Input, Concatenate
from keras.layers import Lambda
from tensorflow.keras.regularizers import L1L2, L1, L2

import functions


def make_gradcam_heatmap(image, used_model, last_conv_name, pred_index):

    grad_model = models.Model(used_model.inputs, [used_model.get_layer(last_conv_name).output, used_model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_ouput, preds = grad_model(image)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_ouput)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    last_conv_layer_ouput = last_conv_layer_ouput[0]
    heatmap = last_conv_layer_ouput @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    return heatmap




# solo models
def cnn_model(image_shape=(150 , 150, 1), is_feature=False):
    # konfigurace vrstev
    num_conv_filters = 32        # pocet conv. filtru
    max_pool_size = (2, 2)       # velikost maxpool filtru
    conv_kernel_size = (3, 3)    # velikost conv. filtru
    imag_shape = image_shape     # vlastnosti obrazku
    dropout_prob = 0.25          # pravdepodobnost odstraneni neuronu

    model = Sequential()  # Typ modelu
    # 1. vrstva
    model.add(Conv2D(filters=num_conv_filters, kernel_size=(conv_kernel_size), input_shape=imag_shape,
                     activation='relu', kernel_regularizer=L1L2(l1=0.1e-4, l2=0.1e-5)#1,data_format='channels_last'
                     #bias_regularizer=L1(l1=0.01), activity_regularizer=L2(l2=0.01)
                     ))
    model.add(MaxPooling2D(pool_size=max_pool_size))
    model.add(Dropout(dropout_prob))

    # 2. vrstva
    model.add(Conv2D(filters=num_conv_filters * 2, kernel_size=(conv_kernel_size), input_shape=imag_shape,
                     activation='relu', kernel_regularizer=L1L2(l1=0.1e-4, l2=0.1e-5)#,data_format='channels_last'
                     #bias_regularizer=L1(l1=0.01), activity_regularizer=L2(l2=0.01)
                     ))
    model.add(MaxPooling2D(pool_size=max_pool_size))
    model.add(Dropout(dropout_prob))
    # 3. vrstva
    model.add(Conv2D(filters=num_conv_filters * 4, kernel_size=(conv_kernel_size), input_shape=imag_shape,
                     activation='relu', kernel_regularizer=L1L2(l1=0.1e-4, l2=0.1e-5)#,data_format='channels_last'
                     #bias_regularizer=L1(l1=0.01), activity_regularizer=L2(l2=0.01)
                     ))
    model.add(MaxPooling2D(pool_size=max_pool_size))
    model.add(Dropout(dropout_prob))


    # Plne propojena vrstva
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dropout_prob*2))
    # model.add(Dense(128, activation="relu"))
    # odstraneni neuronu proti overfittingu
    if not is_feature:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

#Local X normal features
def cnn_local_features(image_shape=(15,15,1)):
    num_conv_filters = 16  # pocet conv. filtru
    max_pool_size = (2, 2)  # velikost maxpool filtru
    conv_kernel_size = (3, 3)  # velikost conv. filtru
    imag_shape = image_shape  # vlastnosti obrazku
    dropout_prob = 0.25  # pravdepodobnost odstraneni neuronu
    model = Sequential()
    #Layer 1
    model.add(Conv2D(filters=num_conv_filters, kernel_size=conv_kernel_size, input_shape=imag_shape, activation="relu"))
    model.add(MaxPool2D(max_pool_size))
    model.add(Dropout(dropout_prob))
    #Layer 2
    model.add(Conv2D(filters=num_conv_filters, kernel_size=conv_kernel_size, activation="relu"))
    model.add(MaxPool2D(max_pool_size))
    model.add(Dropout(dropout_prob))
    #Connected Layer
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    print(model.summary())
    return model

def cnn_feature_model(image_shape=(150,150,1), feature_shape=None, feature_type=None):
    image = Input(shape=(image_shape), name="image")

    if feature_type == "local_solo":
        feature = Input(shape=feature_shape, name=f'patch_input_image')
        cnn_base_local_network = cnn_local_features(image_shape=(feature_shape[1], feature_shape[2], feature_shape[3]))
        local_patch_outputs_image = []
        for i in range(feature_shape[0]):
            local_patch_outputs_image.append(cnn_base_local_network(feature[:, i, :, :, :]))
        concat_patch = Concatenate(axis=1)(local_patch_outputs_image)  # for patch
        dense = Dense(128, activation="relu")(concat_patch)
        output = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=feature, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    cnn_base_network = cnn_model(image_shape, is_feature=True)
    image_output = cnn_base_network(image)

    if feature_type == "strokes":
        feature = Input(shape=(1,), name="strokes")
        concat = Concatenate()([image_output, feature])
    if feature_type == "histogram":
        feature = Input(shape=feature_shape, name="histogram")
        concat = Concatenate()([image_output, feature])
    if feature_type == "wavelet":
        feature = Input(shape=feature_shape, name="wavelet")
        dense_wavelet = Dense(128, activation="relu", name="dense_feat1")(feature)
        concat = Concatenate()([image_output, dense_wavelet])
    if feature_type == "tri_shape":
        feature = Input(shape=(6,), name="tri_shape")
        concat = Concatenate()([image_output, feature])
    if feature_type == "tri_surface":
        feature = Input(shape=(3,), name="tri_surface")
        concat = Concatenate()([image_output, feature])
    if feature_type == "six_fold":
        feature = Input(shape=(18,), name="six_fold")
        concat = Concatenate()([image_output, feature])
    if feature_type == "local":
        feature = Input(shape=feature_shape, name=f'patch_input_image')
        cnn_base_local_network = cnn_local_features(image_shape=(feature_shape[1], feature_shape[2], feature_shape[3]))
        local_patch_outputs_image = []
        for i in range(feature_shape[0]):
            local_patch_outputs_image.append(cnn_base_local_network(feature[:, i, :, :, :]))
        concat_patch = Concatenate(axis=1)(local_patch_outputs_image)  # for patch
        dense_patch = Dense(128, activation="relu")(concat_patch)
        concat = Concatenate()([image_output, dense_patch])


    output = Dense(1, activation='sigmoid', name="clasificator")(concat)
    model = Model(inputs=[image, feature], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model



def snn_base_cnn_model(image_shape=(100 , 100, 1)):
    num_conv_filters = 32  # pocet conv. filtru
    max_pool_size = (2, 2)  # velikost maxpool filtru
    conv_kernel_size = (3, 3)  # velikost conv. filtru
    imag_shape = image_shape  # vlastnosti obrazku
    dropout_prob = 0.25  # pravdepodobnost odstraneni neuronu
    model = Sequential()


    model.add(Conv2D(filters=num_conv_filters, kernel_size=(conv_kernel_size[0]*2, conv_kernel_size[1]*2),
                        input_shape=imag_shape, activation='relu', data_format='channels_last',
                     #kernel_regularizer=L1L2(l1=0.1e-4, l2=0.1e-5)
                     ))
    model.add(BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9))
    model.add(MaxPool2D(pool_size=max_pool_size))
    model.add(Dropout(dropout_prob))


    model.add(Conv2D(filters=num_conv_filters*2, kernel_size=(conv_kernel_size), input_shape=imag_shape,
                        activation='relu', data_format='channels_last'#, kernel_regularizer=L1L2(l1=0.1e-4, l2=0.1e-5)
                     ))
    model.add(BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9))
    model.add(MaxPool2D(pool_size=max_pool_size))
    model.add(Dropout(dropout_prob))


    model.add(Conv2D(filters=num_conv_filters*3, kernel_size=(conv_kernel_size), input_shape=imag_shape,
                        activation='relu', data_format='channels_last'#, kernel_regularizer=L1L2(l1=0.1e-4, l2=0.1e-5)
                     , name="last_conv"))
    model.add(MaxPool2D(pool_size=max_pool_size))
    model.add(Dropout(dropout_prob))

    model.add(Flatten())
    model.add(Dense(512 #, kernel_regularizer=L2(l2=0.1e-5)
                    , activation='relu'))
    model.add(Dropout(dropout_prob*2))

    model.add(Dense(128, activation='relu'))

    # model.add(Conv2D(filters=96, kernel_size=(11, 11), activation='relu', name='conv1_1', strides=4, input_shape=imag_shape), padding='same')
    # model.add(BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9))
    # model.add(MaxPooling2D((3,3), strides=(2, 2)))
    # model.add(ZeroPadding2D((2, 2)))

    # model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu', name='conv2_1', strides=1))
    # model.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
    # model.add(MaxPooling2D((3,3), strides=(2, 2)))
    # model.add(Dropout(0.3))
    # model.add(ZeroPadding2D((1, 1)))
    #
    # model.add(Conv2D(filters=384, kernel_size=(3, 3), activation='relu', name='conv3_1', strides=1))
    # model.add(ZeroPadding2D((1, 1)))
    #
    # model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv3_2', strides=1))
    # model.add(MaxPooling2D((3,3), strides=(2, 2)))
    # model.add(Dropout(0.3))
    # model.add(Flatten(name='flatten'))
    # model.add(Dense(1024, W_regularizer=L2(l2=0.0005), activation='relu'))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(128, W_regularizer=L2(l2=0.0005), activation='relu'))
    model.summary()
    return model

def snn_model(image_shape=(100, 100, 1), feature_shape=None, feature_type=None):

    if feature_type == "local_solo":
        feature1 = Input(shape=feature_shape, name=f'patch_input_image1_1')
        feature2 = Input(shape=feature_shape, name=f'patch_input_image2_2')
        cnn_base_local_network1 = cnn_local_features(image_shape=(feature_shape[1], feature_shape[2], feature_shape[3]))
        cnn_base_local_network2 = cnn_local_features(image_shape=(feature_shape[1], feature_shape[2], feature_shape[3]))
        local_patch_outputs_image1 = []
        local_patch_outputs_image2 = []
        for i in range(feature_shape[0]):
            local_patch_outputs_image1.append(cnn_base_local_network1(feature1[:, i, :, :, :]))
        for i in range(feature_shape[0]):
            local_patch_outputs_image2.append(cnn_base_local_network2(feature2[:, i, :, :, :]))
        concat_img1 = Concatenate(axis=1)(local_patch_outputs_image1)  # for patch 1
        concat_img2 = Concatenate(axis=1)(local_patch_outputs_image2)  # for patch 2
        concat = Concatenate()([concat_img1, concat_img2])
        dense = Dense(128, activation="relu")(concat)
        output = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=[feature1, feature2], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    base_network = snn_base_cnn_model(image_shape)
    image1 = Input(shape=(image_shape), name="image1")
    print(f'\nshape of im1 is {image1.shape}')
    image2 = Input(shape=(image_shape), name="image2")
    print(f'\nshape of im2 is {image2.shape}')

    # Nahrání obrázků a předzpracování skrze CNN
    preprocessed_image1 = base_network(image1)
    print(preprocessed_image1.shape)
    preprocessed_image2 = base_network(image2)
    print(preprocessed_image2.shape)

    if feature_type is None:
        concat = Concatenate()([preprocessed_image1, preprocessed_image2])
        dense = Dense(128, activation='relu')(concat)
        output = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=[image1, image2], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    image_distance = Lambda(functions.euclidan_distance,
                            output_shape=functions.euclidan_dist_output_shape)([preprocessed_image1, preprocessed_image2])

    if feature_type == "strokes":
        feature1 = Input(shape=(1,), name='feature1')
        feature2 = Input(shape=(1,), name='feature2')
        concat = Concatenate()([preprocessed_image1, preprocessed_image2, feature1, feature2])
    if feature_type == "histogram":
        feature1 = Input(shape=(feature_shape,), name="feature1")
        feature2 = Input(shape=(feature_shape,), name="feature2")
        concat = Concatenate()([preprocessed_image1, preprocessed_image2, feature1, feature2])
    if feature_type == "wavelet":
        feature1 = Input(shape=(feature_shape,), name="feature1")
        feature2 = Input(shape=(feature_shape,), name="feature2")
        dense_wavelet1 = Dense(128, activation="relu", name="dense_feat1")(feature1)
        dense_wavelet2 = Dense(128, activation="relu", name="dense_feat2")(feature2)
        concat = Concatenate()([image_distance, dense_wavelet1, dense_wavelet2])
    if feature_type == "tri_shape":
        feature1 = Input(shape=(6,), name="feature1")
        feature2 = Input(shape=(6,), name="feature2")
        concat = Concatenate()([preprocessed_image1, preprocessed_image2, feature1, feature2])
    if feature_type == "tri_surface":
        feature1 = Input(shape=(3,), name="feature1")
        feature2 = Input(shape=(3,), name="feature2")
        concat = Concatenate()([preprocessed_image1, preprocessed_image2, feature1, feature2]) #TODO OTESTOVAT A ZMENIT
    if feature_type == "six_fold":
        feature1 = Input(shape=(18), name="feature1")
        feature2 = Input(shape=(18), name="feature2")
        concat = Concatenate()([image_distance, feature1, feature2])

    #Pro užití lokálních příznaků
    elif feature_type == "local":
        feature1 = Input(shape=feature_shape, name=f'patch_input_image1')
        feature2 = Input(shape=feature_shape, name=f'patch_input_image2')

        cnn_base_local_network1 = cnn_local_features(image_shape=(feature_shape[1], feature_shape[2], feature_shape[3]))
        cnn_base_local_network2 = cnn_local_features(image_shape=(feature_shape[1], feature_shape[2], feature_shape[3]))
        local_patch_outputs_image1 = []
        local_patch_outputs_image2 = []
        for i in range(feature_shape[0]):
            local_patch_outputs_image1.append(cnn_base_local_network1(feature1[:, i, :, :, :]))
        for i in range(feature_shape[0]):
            local_patch_outputs_image2.append(cnn_base_local_network2(feature2[:, i, :, :, :]))
        concat_img1 = Concatenate(axis=1)(local_patch_outputs_image1) #for patch 1
        concat_img2 = Concatenate(axis=1)(local_patch_outputs_image2) #for patch 2

        concat_img1 = Concatenate()([preprocessed_image1, concat_img1])
        concat_img2 = Concatenate()([preprocessed_image2, concat_img2])
        concat = Concatenate()([concat_img1, concat_img2])

    output = Dense(1, activation='sigmoid')(concat)
    model = Model(inputs=[image1, feature1, image2, feature2], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model
