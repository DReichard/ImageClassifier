import keras as K
from keras import *
import numpy
from keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Activation, BatchNormalization, AveragePooling2D, \
    Lambda, ReLU, Add
from keras.optimizers import Adam, Adamax
from assembling_scripts.srm_matrices import get_kernels
from keras.utils.conv_utils import convert_kernel


conv_init = K.initializers.Constant(value=0.1)


def type_1_layer(x, n):
    x = Conv2D(n,
               (3, 3),
               padding="same",
               use_bias=True, bias_initializer=conv_init,
               kernel_initializer='random_normal',
               data_format="channels_first")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def type_2_layer(x, n):
    y = type_1_layer(x, n)
    y = Conv2D(n, kernel_size=(3, 3), padding='same',
               kernel_initializer='random_normal',
               use_bias=True, bias_initializer=conv_init,
               data_format="channels_first")(y)
    y = BatchNormalization()(y)
    x = Add()([x, y])
    return x


def type_3_layer(x, n):
    z = type_1_layer(x, n)
    z = Conv2D(n, kernel_size=(3, 3), padding='same',
                               kernel_initializer='random_normal',
                               use_bias=True, bias_initializer=conv_init,
                               data_format="channels_first")(z)
    z = BatchNormalization()(z)
    z = AveragePooling2D((3, 3),
                         (2, 2),
                         padding='same',
                         data_format="channels_first")(z)
    y = Conv2D(n, data_format="channels_first",
                        kernel_size=(1, 1), strides=(2, 2),
                        use_bias=True, bias_initializer=conv_init,
                        padding='same')(x)
    y = BatchNormalization()(y)
    t = Add()([z, y])
    return t


def type_4_layer(x, n):
    x = type_1_layer(x, n)
    x = Conv2D(n,
               data_format="channels_first", kernel_size=(3, 3),
               padding='same',
               use_bias=True, bias_initializer=conv_init,
               kernel_initializer='random_normal')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D(data_format="channels_first")(x)
    return x

def assemble_network_xu_mod(n):
    print("Assembling start")

    input_value = Input(shape=(1, 128, 128))
    x = type_1_layer(input_value, 64)
    x = type_1_layer(x, 16)

    x = type_2_layer(x, 16)
    x = type_2_layer(x, 16)
    x = type_2_layer(x, 16)

    x = type_3_layer(x, 16)
    x = type_3_layer(x, 64)

    # x = AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first")(x)
    # x = type_1_layer(x, 128)
    #
    # x = GlobalAveragePooling2D(data_format="channels_first")(x)
    x = type_4_layer(x, 128)

    x = Dense(128, activation='relu')(x)
    x = Dense(2)(x)
    out = Activation('softmax')(x)
    network = Model(inputs=input_value, outputs=out)
    opt = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)
    network.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'], )
    print("Assembly done")
    return network


