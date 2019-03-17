from keras import models
from keras import layers
from keras.layers.normalization import BatchNormalization


def add_second_type(network, size):
    print('WIP')


def add_first_type(network, size):
    network.add(layers.Conv2D(size, kernel_size=(3, 3), input_shape=(size*4, size*4, 1)))
    network.add(BatchNormalization())
    network.add(layers.ReLU())


def get_network():
    network = models.Sequential()
    add_first_type(network, 64)
    add_first_type(network, 16)
    # add_first_type(network, 16)
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    print('Network Compiled')
    return network

