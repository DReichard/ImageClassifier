import glob

from keras import models
from keras import layers
import numpy as np
from keras.callbacks import LambdaCallback
from keras.utils import np_utils
from matplotlib.pyplot import imshow
from skimage.util import view_as_blocks
from sklearn.model_selection import train_test_split

import image_parsing
import prepare_dataset
import rdn_assembly
import matplotlib as plt
from PIL import Image
from scipy import misc, ndimage, signal

clean_path = 'D:\\nir_datasets\\jpg\\clean\\clean_cut_images2\\*'
affected_path = 'D:\\nir_datasets\\jpg\\affected\\sorted\\vsl_lsb_5kb\\2019-03-17-15-04-40-207\\input000\\*';
dataset_limit = 1000
n=128

def load_images(path_pattern):

    files=glob.glob(path_pattern)

    X=[]
    indx = 0
    for f in files[0:1000]:
        I = misc.imread(f)
        indx = indx + 1
        if indx % 1000 == 0:
            print(indx, 'files loaded')
        patches = view_as_blocks(I, (n, n))
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                X.append( [ patches[i,j] ] )
    X = np.array(X)
    return X

images_clean = load_images(clean_path)
images_aff = load_images(affected_path)

labels_clean = [0] * len(images_clean)
labels_aff = [1] * len(images_aff)

c = np.empty((images_clean.shape[0] + images_aff.shape[0], images_clean.shape[1], images_clean.shape[2], images_clean.shape[3]), dtype=images_clean.dtype)
c[0::2, :, :, :] = images_clean
c[1::2, :, :, :] = images_aff

d = np.empty((len(labels_clean) + len(labels_aff)), dtype=images_clean.dtype)
d[0::2] = labels_clean
d[1::2] = labels_aff

dataset_data = c.astype('float32')/255
dataset_labels = d
dataset_labels = np_utils.to_categorical(dataset_labels, 2)
data_train, data_test, labels_train, labels_test = train_test_split(dataset_data, dataset_labels, test_size=0.33, shuffle=False)

print(labels_test)
network = rdn_assembly.get_network2()
print('Fitting...')

network.summary()
network.fit(data_train, labels_train, epochs=4, batch_size=4)

print('Testing...')
test_loss, test_acc = network.evaluate(data_test, labels_test)
print('Test loss: ', test_loss)
print('Test accuracy: ', test_acc)
