import glob

from keras.utils import np_utils
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


def load_images(path_pattern, n):
    files=glob.glob(path_pattern)
    X = []
    indx = 0
    for f in files[0:250]:
        I = misc.imread(f)
        indx = indx + 1
        if indx % 1000 == 0:
            print(indx, 'files loaded')
        patches = view_as_blocks(I, (n, n))
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                X.append( [ patches[i,j] ] )

    X = numpy.array(X)
    return X


def calc_rs(img):
    arr = img[0, :, :]
    res = rs(arr)
    print(res)
    return res


if __name__ == '__main__':
    KB.clear_session()
    print("Loading model")
    from keras.models import load_model
    network = load_model(r'C:\Users\stani\PycharmProjects\ImageClassifier\models\model_xu_2020_j_10full.h5')
    test_cover_path = r"C:\datasets\j-uniward_128\test\cover\*"
    test_stego_path = r"C:\datasets\j-uniward_128\test\stego_3_0\*"
    image_size = 128

    print("Loading validation cover")
    Yc = load_images(test_cover_path, image_size)
    print("Loading validation affected")
    Ys = load_images(test_stego_path, image_size)
    print("Images loaded")
    Y = numpy.vstack((Yc, Ys))

    Yt = numpy.hstack(([0]*len(Yc), [1]*len(Ys)))

    # Yt = np_utils.to_categorical(Yt, 2)

    idx = range(len(Y))
    random.shuffle(idx)

    Y = Y[idx]
    Yt = Yt[idx]

    print("Predicting")

    y_pred_keras = network.predict(Y)
    y_pred_keras = y_pred_keras[:, 1]
    print("Evaluating")
    print(accuracy_score(Yt, y_pred_keras.round()))
    # with Pool(processes=6) as pool:
    #     y_pred_rs = list(pool.map(calc_rs, Y))

    print("Plotting")

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Yt, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    #
    # fpr_rs, tpr_rs, thresholds_rs = roc_curve(Yt, y_pred_rs)
    # auc_rs = auc(fpr_rs, tpr_rs)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='0.8bpp (AUC = {:.3f})'.format(auc_keras))
    # plt.plot(fpr_rs, tpr_rs, label='RS-модуль (AUC = {:.3f})'.format(auc_rs))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC (S-UNIWARD @ 2 densities)')
    plt.legend(loc='best')
    plt.show()
    KB.clear_session()
