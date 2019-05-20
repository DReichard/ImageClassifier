import keras
import numpy
from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, Lambda, Activation, AveragePooling2D, GlobalAveragePooling2D, \
    Dense, K
import tensorflow as tf
from assembling_scripts.srm_matrices import get_kernels
from keras.utils.conv_utils import convert_kernel


def assemble_network_xu(n):
    print("Assembling start")
    model = Sequential()
    # n = 256
    F0 = numpy.array(
       [[-1,  2,  -2,  2, -1],
        [ 2, -6,   8, -6,  2],
        [-2,  8, -12,  8, -2],
        [ 2, -6,   8, -6,  2],
        [-1,  2,  -2,  2, -1]])
    F = numpy.reshape(F0, (F0.shape[0],F0.shape[1],1,1) )
    bias=numpy.array([0])
    # print(F.shape)

    model.add(Conv2D(1, (5,5), padding="same", data_format="channels_first", input_shape=(1,n,n), activation='relu', weights=[F, bias]))

    model.add(Conv2D(8, (5,5), padding="same", data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Lambda(lambda x: K.abs(x)))
    model.add(Activation("tanh"))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first"))

    model.add(Conv2D(16, (5,5), padding="same", data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first"))

    model.add(Conv2D(32, (1,1), padding="same", data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first"))

    model.add(Conv2D(64, (1,1), padding="same", data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first"))

    model.add(Conv2D(128, (1,1), padding="same", data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(GlobalAveragePooling2D(data_format="channels_first"))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    opt = keras.optimizers.Adam(lr=0.004, decay=0.000002, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'], )
    print("Assembly done")
    return model
