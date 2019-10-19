import glob
import json
from scipy import misc
import numpy
import random
from skimage.util.shape import view_as_blocks
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from rs_analysis import rs
from sklearn.metrics import accuracy_score
import keras as K
from keras import backend as KB
from keras.models import load_model
import base64
import json
import cv2


def fwriteKeyVals(data, f, indent=0):
    if isinstance(data, list):
        f.write( "\n" + "    " * indent + "[" )
        for i in range(len(data) ):
            if ( i == 0):
                f.write( "[" )
            else:
                f.write( "    " * indent + " [" )
            for j in range(len(data[0])):
                f.write( "%3d" % data[i][j] )
                f.write( "," ) if j != len(data[0])-1 else (f.write( "]," ) if i != len(data)-1 else f.write( "]" ))
            f.write( "\n" ) if i != len(data)-1 else f.write( "]" )
    elif isinstance(data, dict):
        f.write( "\n" + "    " * indent + "{" )
        for k, v in data.iteritems():
            f.write( "\n" + "    " * indent + "\"" + k + "\"" + ": ")
            fwriteKeyVals(v, f, indent + 1)
            if data.keys()[-1] != k:
                 f.write( "," )
        f.write( "\n" + "    " * indent + "}" )
    else:
        f.write("\"" + data + "\"")

def load_images(path_pattern, n):
    files=glob.glob(path_pattern)
    X = []
    indx = 0
    for f in files[0:250]:
        I = cv2.imread(f, 0)
        indx = indx + 1
        if indx % 1000 == 0:
            print(indx, 'files loaded')
        patches = view_as_blocks(I, (n, n))
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                X.append( [ patches[i,j] ] )

    f = numpy.float32(X)
    return numpy.array(f)


KB.clear_session()
print("Loading model")

target_path = r"D:\diploma\gallery\test\4.jpg"
Y = load_images(target_path, 128)
js = json.dumps(Y.flatten().tolist())
network = load_model(r'C:\Users\stani\PycharmProjects\ImageClassifier\models\model_xu_2020_j_float32_undiv_1full.h5')
y_pred_keras = network.predict(Y)
#
print(Y)
print("Result:")
print(y_pred_keras)
