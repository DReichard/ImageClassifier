import glob
import numpy
from keras.callbacks import LearningRateScheduler
from scipy import misc
import onnxmltools
from skimage.util import view_as_blocks
from assembling_scripts.cnn_xu_assembly import assemble_network_xu
from assembling_scripts.cnn_xu_mod_assembly import assemble_network_xu_mod
from assembling_scripts.rdn_assembly import get_network
from training_scripts.batch_generator import pair_generator
import json
from keras import backend as KB


def load_images(path_pattern):
    # Crop squared blocks of the image with size:
    n = 128
    files=glob.glob(path_pattern)
    X=[]
    indx = 0
    for f in files[0:10000]:
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
# train_cover_path = "D:\\nir_datasets\\ready_datasets\\wow_64\\train\\cover"
# train_stego_path = "D:\\nir_datasets\\ready_datasets\\wow_64\\train\\stego"
# train_cover_path = "C:\\datasets\\wow_64\\train\\cover"
# train_stego_path = "C:\\datasets\\wow_64\\train\\stego"

test_cover_path = r"C:\datasets\j-uniward_128_2020\test\cover"
test_stego_path = r"C:\datasets\j-uniward_128_2020\test\stego"
# test_cover_path = "D:\\nir_datasets\\ready_datasets\\wow_64\\test\\cover"
# test_stego_path = "D:\\nir_datasets\\ready_datasets\\wow_64\\test\\stego"
# test_cover_path = "C:\\datasets\\wow_64\\test\\cover"
# test_stego_path = "C:\\datasets\\wow_64\\test\\stego"


image_size = 128
batch_size = 30
train_set_size = 30000
test_set_size = 500
lr_epoch_threshold = 400
lr_old_val = 0.001
lr_new_val = 0.0001

train_epoch_size = train_set_size / batch_size
test_epoch_size = test_set_size /batch_size
train_generator = pair_generator(train_cover_path, train_stego_path, image_size, batch_size, train_set_size, True)
test_generator = pair_generator(test_cover_path, test_stego_path, image_size, batch_size, test_set_size, True)
network = assemble_network_xu_mod(image_size)


def scheduler(epoch):
    if epoch >= lr_epoch_threshold:
        return lr_new_val
    return lr_old_val


change_lr = LearningRateScheduler(scheduler)

ep = 1
i = 0
# #
# from keras.models import load_model
# network = load_model('../models/model_xu_2020_j_' + str(i) + 'full.h5')
history_tr_acc = []
history_val_acc = []
history_tr_loss = []
history_val_loss = []
while True:
    i += ep
    print(str(i))
    history = network.fit_generator(
        train_generator,
        epochs=ep,
        steps_per_epoch=train_epoch_size,
        validation_data=test_generator,
        validation_steps=test_epoch_size,
        callbacks=[change_lr],
        verbose=2)

    open("../models/model_xu_2020_j_"+str(i)+".json", 'w').write(network.to_json())
    network.save_weights("../models/model_xu_2020_j_"+str(i)+".h5")
    network.save("../models/model_xu_2020_j_"+str(i)+"full.h5")
    # onnx_model = onnxmltools.convert_keras(network, target_opset=7)
    # onnxmltools.utils.save_text(onnx_model, '../models/model_xu_' + str(i) + '_onnx.json')
    # onnxmltools.utils.save_model(onnx_model, '../models/model_xu_' + str(i) + '_onnx.onnx')

    history_tr_acc.append(history.history['acc'])
    history_val_acc.append(history.history['val_acc'])
    history_tr_loss.append(history.history['loss'])
    history_val_loss.append(history.history['val_loss'])

    file = '../models/tr_acc2.json'
    with open(file, 'w') as filetowrite:
        filetowrite.write(json.dumps(history_tr_acc))
    file = '../models/val_acc2.json'
    with open(file, 'w') as filetowrite:
        filetowrite.write(json.dumps(history_val_acc))
    file = '../models/tr_loss2.json'
    with open(file, 'w') as filetowrite:
        filetowrite.write(json.dumps(history_tr_loss))
    file = '../models/val_loss2.json'
    with open(file, 'w') as filetowrite:
        filetowrite.write(json.dumps(history_val_loss))

    import matplotlib.pyplot as plt
    # Plot training & validation accuracy values
    plt.plot(history_tr_acc)
    plt.plot(history_val_acc)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    # Plot training & validation loss values
    plt.plot(history_tr_loss)
    plt.plot(history_val_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    KB.clear_session()


#13 норм

