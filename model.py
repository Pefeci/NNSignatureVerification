"""
@author: Petr Čírtek
"""
# Not used imports
import keras
import PIL
from keras import backend
from keras import layers, models
import tensorflow as tf
from keras.backend import relu, sigmoid
# True imports
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D, \
    BatchNormalization, ZeroPadding2D, Input, Concatenate, Reshape
from keras.layers import Lambda
from tensorflow.keras.regularizers import L1L2, L1, L2

import functions
from functions import euclidan_distance, euclidan_dist_output_shape
from keras.optimizers import SGD, RMSprop, Adadelta

# gradcam model:
# def gradcam_heatmap(image, used_model, layer_name, positive_class=True, preprocess_input_function=None, normalize=True):
#     # preprocess may be redundant /TODO check its all wrong?
#     image = tf.convert_to_tensor(image)
#     if preprocess_input_function is not None:
#         image = preprocess_input_function(image)
#
#     with tf.GradientTape() as tape:
#         tape.watch(image)
#
#         predictions = used_model(image, training=False)
#         if(positive_class):
#             target_class_output = predictions[:,1]
#         else:
#             target_class_output = predictions[:,0]
#
#     last_conv_layer_output = used_model.get_layer(layer_name).output
#     gradients = tape.gradient(target_class_output,last_conv_layer_output)
#
#     pooled_gradient = tf.reduce_mean(gradients, axis=(0,1,2))
#     heatmap = tf.reduce_sum(tf.multiply(pooled_gradient, last_conv_layer_output), axis=-1)
#     heatmap = tf.nn.relu(heatmap)
#
#     if normalize:
#         min_value = tf.reduce_min(heatmap)
#         max_value = tf.reduce_max(heatmap)
#         heatmap = (heatmap - min_value)/(max_value - min_value)
#
#     heatmap = heatmap.numpy()
#     return heatmap



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
def cnn_model(image_shape=(150 , 150, 1)):
    # konfigurace vrstev
    num_conv_filters = 32        # pocet conv. filtru
    max_pool_size = (2, 2)       # velikost maxpool filtru
    conv_kernel_size = (3, 3)    # velikost conv. filtru
    imag_shape = image_shape     # vlastnosti obrazku
    dropout_prob = 0.25          # pravdepodobnost odstraneni neuronu

    # Predspracovani funkce
    rescale = Sequential([
        layers.Rescaling(1./255)
    ])
    threshold = Sequential([
        layers.ThresholdedReLU(theta=0.6)
    ])
    rotate = Sequential([
        layers.RandomRotation(factor=(-0.05, 0.05))
    ])
    translate = Sequential([
        layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1,0.1))
    ])
    zoom = Sequential([
        layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1,0.1))
    ])


    model = Sequential()  # Typ modelu

    #Preprocessed Vrstva
    #model.add(rescale)
    #model.add(threshold)
    #model.add(rotate)
    #model.add(translate)
    #model.add(zoom)

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

    # odstraneni neuronu proti overfittingu
    model.add(Dropout(dropout_prob*2))

    # Vyhodnoceni
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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
    print(model.summary())
    return model


#Local X normal features
def cnn_local_features(image_shape=(100,100,1)):
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
    return model



# TODO delete cause REdundant
# def cnn_procces_local(patch_size=10, image_shape=(100,100,1)):
#     patch_inputs = [Input(shape=image_shape) for i in range(patch_size)]
#     cnn_base_network = cnn_local_features(image_shape=image_shape)
#     patch_outputs = [cnn_base_network(patch_input) for patch_input in patch_inputs]
#     concat = Concatenate()(patch_outputs)
#     dense = Dense(128, activation="relu")(concat)
#     return Model(inputs=patch_inputs, outputs=dense)






def snn_model(image_shape=(100, 100, 1), feature_shape=None, feature_type=None):
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
        concat = Concatenate()([image_distance, feature1, feature2])

    if feature_type == "wavelet":
        feature1 = Input(shape=(feature_shape,), name="feature1")
        feature2 = Input(shape=(feature_shape,), name="feature2")
        dense_wavelet1 = Dense(128, activation="relu", name="dense_feat1")(feature1)
        dense_wavelet2 = Dense(128, activation="relu", name="dense_feat2")(feature2)
        concat = Concatenate()([image_distance, dense_wavelet1, dense_wavelet2])
    if feature_type == "tri_shape":
        feature1 = Input(shape=(6,), name="feature1")
        feature2 = Input(shape=(6,), name="feature2")
        concat = Concatenate()([image_distance, feature1, feature2])
    if feature_type == "tri_surface":
        feature1 = Input(shape=(3,), name="feature1")
        feature2 = Input(shape=(3,), name="feature2")
        concat = Concatenate()([image_distance, feature1, feature2])
    if feature_type == "six_fold":
        feature1 = Input(shape=(18), name="feature1")
        feature2 = Input(shape=(18), name="feature2")
        concat = Concatenate()([image_distance, feature1, feature2])

    #Pro užití lokálních příznaků
    #Define num_patches
    # num_patches = 10
    # patch_inputs_image1 = [Input(shape=image_shape, name=f'patch_input_image1_{i}') for i in range(num_patches)]
    # patch_inputs_image2 = [Input(shape=image_shape, name=f'patch_input_image2_{i}') for i in range(num_patches)]
    # cnn_base_local_network = cnn_local_features(image_shape=image_shape)
    # local_patch_outputs_image1 = [cnn_base_local_network(patch_input) for patch_input in patch_inputs_image1]
    # local_patch_outputs_image2 = [cnn_base_local_network(patch_input) for patch_input in patch_inputs_image2]
    # concat_img1 = Concatenate()(local_patch_outputs_image1) #for patch 1
    # concat_img2 = Concatenate()(local_patch_outputs_image2) #for patch 2
    # concat_img1 = Concatenate()([preprocessed_image1, concat_img1])
    # concat_img2 = Concatenate()([preprocessed_image2, concat_img2])
    # concat = Concatenate()(concat_img1, concat_img2)



    # Určení vzdálenosti od dvou obrázků
    #distance = Lambda(euclidan_distance, output_shape=euclidan_dist_output_shape)([preprocessed_image1, preprocessed_image2])
    #model = Model(inputs=[image1, image2], outputs=distance)
    #rms = RMSprop(learning_rate=1e-4, rho=0.9, epsilon=1e-08)

    # TODO coment this back and forth for features
    #concat = Concatenate()([preprocessed_image1, preprocessed_image2])
    # dense = Dense(128, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(concat)
    #TODO also theres need to change this
    model = Model(inputs=[image1, feature1, image2, feature2], outputs=output)
    #model.compile(loss=functions.contrastive_loss, optimizer=rms, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model



# def debug():
#     Model = cnn_model()
#     Model.build((None,200,200,1))
#     Model.summary()


def snn_model_with_strokes():
    return