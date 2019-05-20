import glob

import numpy
from keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.models import load_model, Model
from scipy import misc
from skimage.util import view_as_blocks


def load_images(path_pattern):

    files=glob.glob(path_pattern)

    X=[]
    indx = 0
    for f in files[0:15000]:
        I = misc.imread(f)
        indx = indx + 1
        if indx % 1000 == 0:
            print(indx, 'files loaded')
        patches = view_as_blocks(I, (256, 256))
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                X.append( [ patches[i,j] ] )
    X=numpy.array(X)
    return X

model = load_model('model_xu_10full.h5')


Yc = load_images('D:\\nir_datasets\\html\\clean\\*')
Ys = load_images('D:\\nir_datasets\\html\\affected\\*')
print(Yc[0][0][0])
res = model.predict(Yc)

numpy.set_printoptions(precision=3)
numpy.set_printoptions(suppress=True)
print(res)
