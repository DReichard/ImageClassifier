from keras import models, Input, Model
from keras import layers
from keras.layers import Conv2D, ReLU, Dense, Add, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adamax, RMSprop


def first_type_layer(input_value, size):
    convolution_layer = Conv2D(size, kernel_size=(3, 3), padding='same', kernel_initializer='random_normal', data_format="channels_first")(input_value)
    batch_normalized_layer = BatchNormalization()(convolution_layer)
    activation_layer = ReLU()(batch_normalized_layer)
    return activation_layer


def second_type_layer(input_value, size):
    first_sublayer = first_type_layer(input_value, size)
    convolution_layer = Conv2D(size, kernel_size=(1, 1), padding='same', kernel_initializer='random_normal', data_format="channels_first")(first_sublayer)
    batch_normalized_layer = BatchNormalization()(convolution_layer)
    add_layer = Add()([batch_normalized_layer, input_value])
    return add_layer


def third_type_layer(input_value, size):
    first_sublayer = first_type_layer(input_value, size)
    convolution_layer = Conv2D(size, kernel_size=(3, 3), padding='same', kernel_initializer='random_normal', data_format="channels_first")(first_sublayer)
    batch_normalized_layer = BatchNormalization()(convolution_layer)
    pooling_layer = AveragePooling2D((3, 3), (2, 2), padding='same', data_format="channels_first")(batch_normalized_layer)
    side_value = Conv2D(size, data_format="channels_first", kernel_size=(1, 1), strides=(2, 2), padding='same')(input_value)
    side_value = BatchNormalization()(side_value)
    add_layer = Add()([pooling_layer, side_value])
    return add_layer


def fourth_type_layer(input_value, size):
    first_sublayer = first_type_layer(input_value, size)
    convolution_layer = Conv2D(size, data_format="channels_first", kernel_size=(1, 1), padding='same', kernel_initializer='random_normal')(first_sublayer)
    batch_normalized_layer = BatchNormalization()(convolution_layer)
    global_average = GlobalAveragePooling2D(data_format="channels_first")(batch_normalized_layer)
    return global_average


def get_network():
    # input_value = Input(shape=(256, 256, 1))
    input_value = Input(shape=(1, 256, 256))
    # x = first_type_layer(input_value, 64)
    x = first_type_layer(input_value, 16)
    # x = first_type_layer(x, 16)

    x = second_type_layer(x, 16)
    x = second_type_layer(x, 16)
    x = second_type_layer(x, 16)
    x = second_type_layer(x, 16)
    x = second_type_layer(x, 16)

    x = third_type_layer(x, 16)
    x = third_type_layer(x, 64)
    x = third_type_layer(x, 128)
    x = third_type_layer(x, 256)
    x = fourth_type_layer(x, 512)
    x = Dense(512, activation='relu')(x)

    output = Dense(2, activation='softmax')(x)
    # print("SUMMARY")
    network = Model(inputs=input_value, outputs=output)
    # network.summary()
    # print("SUMMARY")
    opt = RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
    # opt = Adamax(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    network.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    print('Network Compiled')
    return network
