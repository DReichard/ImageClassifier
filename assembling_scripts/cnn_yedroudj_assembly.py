import keras
import numpy
from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, Lambda, Activation, AveragePooling2D, GlobalAveragePooling2D, \
    Dense, K
import tensorflow as tf
from assembling_scripts.srm_matrices import get_kernels
from keras.utils.conv_utils import convert_kernel


def assemble_network_tedroudj(n):
    print("Assembling start")
    model = Sequential()

    F0 = numpy.array(
        [[-1, 2, -2, 2, -1],
         [2, -6, 8, -6, 2],
         [-2, 8, -12, 8, -2],
         [2, -6, 8, -6, 2],
         [-1, 2, -2, 2, -1]])

    F0 = get_kernels().astype(numpy.float32)

    F = numpy.reshape(F0, (F0.shape[0], F0.shape[1], F0.shape[2], 1))
    F = numpy.moveaxis(F, 0, -1)
    bias = numpy.zeros_like(F)

    def srm(shape, dtype=None):
        return F
    model.add(
        Conv2D(30, (5, 5), padding="same", data_format="channels_first", input_shape=(1, n, n), kernel_initializer=srm))
    model.layers[0].trainable = False

    model.add(Conv2D(30, (5, 5), padding="same", data_format="channels_first", kernel_initializer="glorot_normal"))
    model.add(Lambda(lambda x: K.abs(x)))
    model.add(BatchNormalization())
    model.add(BatchNormalization(scale=True))
    model.add(Activation("tanh"))

    model.add(Conv2D(30, (5, 5), padding="same", data_format="channels_first", kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(BatchNormalization(scale=True))
    model.add(Activation("tanh"))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first"))

    model.add(Conv2D(30, (3, 3), padding="same", data_format="channels_first", kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(BatchNormalization(scale=True))
    model.add(Activation("relu"))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first"))

    model.add(Conv2D(32, (3, 3), padding="same", data_format="channels_first", kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(BatchNormalization(scale=True))
    model.add(Activation("relu"))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first"))

    model.add(Conv2D(64, (3, 3), padding="same", data_format="channels_first", kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(BatchNormalization(scale=True))
    model.add(Activation("relu"))

    model.add(GlobalAveragePooling2D(data_format="channels_first"))

    model.add(Dense(256, kernel_initializer="glorot_normal", activation="relu"))
    model.add(Dense(1024, kernel_initializer="glorot_normal", activation="relu"))
    model.add(Dense(2, kernel_initializer="glorot_normal"))
    model.add(Activation('softmax'))

    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.95, decay=0.0001)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print("Assembly done")
    return model
