import glob
import random
import numpy
from keras.utils import np_utils
from scipy import misc
import onnxmltools
from skimage.util import view_as_blocks
from tensorflow import ConfigProto, InteractiveSession
from itertools import chain
from assembling_scripts.cnn_xu_mod_assembly import assemble_network_xu_mod


def load_images(path_pattern, i):
    n = 128
    files=glob.glob(path_pattern)
    X=[]
    indx = 0
    for f in files[0:i]:
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


train_cover_path = r"C:\datasets\j-uniward_128_2020\train\cover"
train_stego_path = r"C:\datasets\j-uniward_128_2020\train\stego"

test_cover_path = r"C:\datasets\j-uniward_128_2020\test\cover"
test_stego_path = r"C:\datasets\j-uniward_128_2020\test\stego"

image_size = 128
batch_size = 50
train_set_size = 10
test_set_size = 5

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
network = assemble_network_xu_mod2(image_size)
print("Loading cover")
Xc = load_images(train_cover_path + r"\*", train_set_size)
print("Loading affected")
Xs = load_images(train_stego_path + r"\*", train_set_size)
print("Loading validation cover")
Yc = load_images(test_cover_path + r"\*", test_set_size)
print("Loading validation affected")
Ys = load_images(test_stego_path + r"\*", test_set_size)
print("Images loaded")
X = numpy.vstack((Xc, Xs))
Y = numpy.vstack((Yc, Ys))
Xt = numpy.hstack(([0]*len(Xc), [1]*len(Xs)))
Yt = numpy.hstack(([0]*len(Yc), [1]*len(Ys)))
Xt = np_utils.to_categorical(Xt, 2)
Yt = np_utils.to_categorical(Yt, 2)
idx= list(range(len(X)))
idx1= idx[:len(idx)//2]
idx2= idx[len(idx)//2:]
idx = list(chain.from_iterable(zip(idx1, idx2)))
X=X[idx]
Xt=Xt[idx]


ep = 1
i = 0
while True:
    i += ep
    print(str(i))
    network.fit(X, Xt, batch_size=25, epochs=ep, validation_data=(Y, Yt), shuffle=False, verbose=2)
    open("../models/model_xu_2020_j_"+str(i)+".json", 'w').write(network.to_json())
    network.save_weights("../models/model_xu_2020_j_"+str(i)+".h5")
    network.save("../models/model_xu_2020_j_"+str(i)+"full.h5")
