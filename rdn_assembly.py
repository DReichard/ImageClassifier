from keras import models, Input, Model
from keras import layers
from keras.layers import Conv2D, ReLU, Dense
from keras.layers.normalization import BatchNormalization


def add_second_type(network, size):
    print('WIP')


def first_type_layer2(network, size):
    network.add(layers.Conv2D(size, kernel_size=(3, 3), input_shape=(size * 4, size * 4, 1)))
    network.add(BatchNormalization())
    network.add(layers.ReLU())


def first_type_layer(input_value, size):
    convolution_layer = Conv2D(size, kernel_size=(3, 3))(input_value)
    batch_normalized_layer = BatchNormalization(axis=1)(convolution_layer)
    activation_layer = ReLU()(batch_normalized_layer)
    return activation_layer


def get_network():
    input_value = Input(shape=(256, 256))
    x = first_type_layer(input_value, 64)
    x = first_type_layer(x, 16)

    output = Dense(2, activation='softmax')(x)

    network = Model(inputs=input_value, outputs=output)
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    print('Network Compiled')
    return network


def get_network_test():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('Network Compiled')
    return model
