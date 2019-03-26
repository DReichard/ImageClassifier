#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Network presented in the paper:
    Structural Design of Convolutional Neural Networks for Steganalysis.
    Xu, Guanshuo and  Wu, Han-Zhou and Shi, Yun-Qing.
    IEEE Signal Processing Letters, vol. 23, issue 5, pp. 708-712. 05/2016.
"""


import sys
import glob

import onnxmltools
from scipy import misc, ndimage, signal
import numpy
import random
import ntpath
import os
from skimage.util.shape import view_as_blocks, view_as_windows

# from keras.layers import Merge, Lambda, Layer
from keras.layers import Lambda, Layer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, UpSampling2D
from keras.layers.core import Reshape
from keras import optimizers
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras import backend as K

# magic
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)


# Crop squared blocks of the image with size:
n=256

F0 = numpy.array(
   [[-1,  2,  -2,  2, -1],
    [ 2, -6,   8, -6,  2],
    [-2,  8, -12,  8, -2],
    [ 2, -6,   8, -6,  2],
    [-1,  2,  -2,  2, -1]])


# {{{ load_images()
def load_images(path_pattern):

    files=glob.glob(path_pattern)

    X=[]
    indx = 0
    for f in files[0:15000]:
        I = misc.imread(f)
        indx = indx + 1
        if indx % 1000 == 0:
            print(indx, 'files loaded')
        patches = view_as_blocks(I, (n, n))
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                X.append( [ patches[i,j] ] )


    X=numpy.array(X)

    return X
# }}}

#
# if len(sys.argv) < 3:
#    print("%s <train cover dir> <train stego dir> <test cover dir> <test stego dir>\n" % sys.argv[0])
#    sys.exit(0)


# Xc = load_images(sys.argv[1]+'/*.pgm')
# Xs = load_images(sys.argv[2]+'/*.pgm')
# Yc = load_images(sys.argv[3]+'/*.pgm')
# Ys = load_images(sys.argv[4]+'/*.pgm')
print("Loading cover")
Xc = load_images('D:\\nir_datasets\\jpg\\clean\\memes\\memes_train\\*')
print("Loading affected")
Xs = load_images('D:\\nir_datasets\\jpg\\affected\\sorted\\memes_5kb\\train\\*')
print("Loading validation cover")
Yc = load_images('D:\\nir_datasets\\jpg\\clean\\memes\\memes_test\\*')
print("Loading validation affected")
Ys = load_images('D:\\nir_datasets\\jpg\\affected\\sorted\\memes_5kb\\test\\*')
print("Images loaded")
X = numpy.vstack((Xc, Xs))
Y = numpy.vstack((Yc, Ys))


Xt = numpy.hstack(([0]*len(Xc), [1]*len(Xs)))
Yt = numpy.hstack(([0]*len(Yc), [1]*len(Ys)))

Xt = np_utils.to_categorical(Xt, 2)
Yt = np_utils.to_categorical(Yt, 2)

idx=range(len(X))
random.shuffle(idx)

X=X[idx]
Xt=Xt[idx]

print("Assembling")
model = Sequential()

F = numpy.reshape(F0, (F0.shape[0],F0.shape[1],1,1) )
bias=numpy.array([0])


print(F.shape)

model.add(Conv2D(1, (5,5), padding="same", data_format="channels_first", input_shape=(1,n,n), activation='relu', weights=[F, bias]))


# {{{ Group 1
model.add(Conv2D(8, (5,5), padding="same", data_format="channels_first"))
model.add(BatchNormalization())
model.add(Lambda(K.abs))
model.add(Activation("tanh"))

model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first"))
# }}}

# {{{ Group 2
model.add(Conv2D(16, (5,5), padding="same", data_format="channels_first"))
model.add(BatchNormalization())
model.add(Activation("tanh"))

model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first"))
# }}}

# {{{ Group 3
model.add(Conv2D(32, (1,1), padding="same", data_format="channels_first"))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first"))
# }}}

# {{{ Group 4
model.add(Conv2D(64, (1,1), padding="same", data_format="channels_first"))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(AveragePooling2D(pool_size=(5, 5), strides=2, padding="same", data_format="channels_first"))
# }}}

# {{{ Group 5
model.add(Conv2D(128, (1,1), padding="same", data_format="channels_first"))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(GlobalAveragePooling2D(data_format="channels_first"))
# }}}


model.add(Dense(2))
model.add(Activation('softmax'))

import rdn_assembly
# model = rdn_assembly.get_network()

model.compile(loss='binary_crossentropy', optimizer="adamax", metrics=['accuracy'])

print("Compiled")

ep = 10
i = 0
while True:
    i += ep
    model.fit(X, Xt, batch_size=50, epochs=ep, validation_data=(Y, Yt), shuffle=True)
    print("****", i, "epochs done")
    open("model_xu_"+str(i)+".json", 'w').write(model.to_json())
    model.save_weights("model_xu_"+str(i)+".h5")
    model.save("model_xu_"+str(i)+"full.h5")
    onnx_model = onnxmltools.convert_keras(model, target_opset=7)
    onnxmltools.utils.save_text(onnx_model, 'model_xu_10_onnx.json')
    onnxmltools.utils.save_model(onnx_model, 'model_xu_10_onnx.onnx')
