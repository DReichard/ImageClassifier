import glob
from scipy import misc
import numpy
import random
from skimage.util.shape import view_as_blocks
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from rs_analysis import rs
from sklearn.metrics import accuracy_score


def magic_spell():
    # magic
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    )
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    set_session(session)


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
    magic_spell()
    print("Loading model")
    from keras.models import load_model
    network = load_model('./models/best/xu_s-uniward_128px-60p/model_xu_366full.h5')
    network2 = load_model('./models/best/xu_j-uniward_128px_76p/model_xu_262full.h5')
    # test_cover_path = "E:\\html\\clean\\*"
    # test_stego_path = "E:\\html\\j_uniward\\*"
    test_cover_path = "C:\\datasets\\s_uniward_128\\test\\cover\\*"
    test_stego_path = "C:\\datasets\\s_uniward_128\\test\\stego\\*"
    test_stego_path2 = "C:\\datasets\\j-uniward_128\\test\\stego_3_0\\*"
    # test_cover_path = "D:\\nir_datasets\\png\\vsl_in\\*"
    # test_stego_path = "D:\\nir_datasets\\png\\vsl_out\\lsb_128\\input000\\*"
    image_size = 128

    print("Loading validation cover")
    Yc = load_images(test_cover_path, image_size)
    print("Loading validation affected")
    Ys = load_images(test_stego_path, image_size)
    Ys2 = load_images(test_stego_path2, image_size)
    print("Images loaded")
    Y = numpy.vstack((Yc, Ys))
    Y2 = numpy.vstack((Yc, Ys2))

    Yt = numpy.hstack(([0]*len(Yc), [1]*len(Ys)))
    Yt2 = numpy.hstack(([0]*len(Yc), [1]*len(Ys2)))

    # Yt = np_utils.to_categorical(Yt, 2)

    idx = range(len(Y))
    # random.shuffle(idx)

    Y = Y[idx]
    Y2 = Y2[idx]
    Yt = Yt[idx]
    Yt2 = Yt2[idx]

    print("Predicting")

    y_pred_keras = network.predict(Y)
    y_pred_keras = y_pred_keras[:, 1]
    y_pred_keras2 = network2.predict(Y2)
    y_pred_keras2 = y_pred_keras2[:, 1]
    print("Evaluating")
    print(accuracy_score(Yt, y_pred_keras.round()))
    print(accuracy_score(Yt, y_pred_keras2.round()))
    # with Pool(processes=6) as pool:
    #     y_pred_rs = list(pool.map(calc_rs, Y))

    print("Plotting")

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Yt, y_pred_keras)
    fpr_keras2, tpr_keras2, thresholds_keras2 = roc_curve(Yt2, y_pred_keras2)
    auc_keras = auc(fpr_keras, tpr_keras)
    auc_keras2 = auc(fpr_keras2, tpr_keras2)
    #
    # fpr_rs, tpr_rs, thresholds_rs = roc_curve(Yt, y_pred_rs)
    # auc_rs = auc(fpr_rs, tpr_rs)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='0.8bpp (AUC = {:.3f})'.format(auc_keras))
    plt.plot(fpr_keras2, tpr_keras2, label='1.5bpp (AUC = {:.3f})'.format(auc_keras2))
    # plt.plot(fpr_rs, tpr_rs, label='RS-модуль (AUC = {:.3f})'.format(auc_rs))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC (S-UNIWARD @ 2 densities)')
    plt.legend(loc='best')
    plt.show()
