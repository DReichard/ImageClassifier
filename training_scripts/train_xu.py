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
import matplotlib.pyplot as plt
from keras.models import load_model
import os.path
from keras.callbacks import Callback
import tensorflow as tf



train_cover_path = r"C:\datasets\j-uniward_128_2020\train\cover"
train_stego_path = r"C:\datasets\j-uniward_128_2020\train\stego"

test_cover_path = r"C:\datasets\j-uniward_128_2020\test\cover"
test_stego_path = r"C:\datasets\j-uniward_128_2020\test\stego"

model_name = "model_xu_2020_j_float32_undiv"
image_size = 128
batch_size = 30
train_set_size = 1200000
test_set_size = 1000
lr_epoch_threshold = 18
lr_old_val = 0.001
lr_new_val = 0.0001

i = 1


class WeightsSaver(Callback):
    def __init__(self, N):
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            print('saving to: ../models/' + model_name + str(i)+ '_' + str(int(self.batch/self.N)) + ".h5")
            self.model.save_weights('../models/' + model_name + str(i)+ '_' + str(int(self.batch/self.N)) + ".h5")
        self.batch += 1

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
model_name = model_name + '_'
KB.clear_session()
train_epoch_size = train_set_size / batch_size
test_epoch_size = test_set_size /batch_size
train_generator = pair_generator(train_cover_path, train_stego_path, image_size, batch_size, train_set_size, True)
test_generator = pair_generator(test_cover_path, test_stego_path, image_size, batch_size, test_set_size, True)
network = assemble_network_xu_mod(image_size)


def scheduler(epoch):
    if i >= lr_epoch_threshold:
        return lr_new_val
    return lr_old_val


ep = 1
change_lr = LearningRateScheduler(scheduler)
if i != 0:
    network = load_model('../models/' + model_name + str(i) + 'full.h5')
while True:
    i += ep
    print(str(i))
    history = network.fit_generator(
        train_generator,
        epochs=1,
        steps_per_epoch=train_epoch_size,
        validation_data=test_generator,
        validation_steps=test_epoch_size,
        callbacks=[change_lr, WeightsSaver(1000)],
        verbose=1)

    open('../models/' + model_name + str(i) + ".json", 'w').write(network.to_json())
    network.save_weights('../models/' + model_name + str(i)+".h5")
    network.save('../models/' + model_name + str(i)+"full.h5")
    # onnx_model = onnxmltools.convert_keras(network, target_opset=7)
    # onnxmltools.utils.save_text(onnx_model, '../models/model_xu_' + str(i) + '_onnx.json')
    # onnxmltools.utils.save_model(onnx_model, '../models/model_xu_' + str(i) + '_onnx.onnx')

    file = '../models/' + model_name + 'tr_acc.json'
    if os.path.isfile(file):
        with open(file, 'r') as f:
            history_tr_acc = json.load(f)
    else:
        history_tr_acc = []
    history_tr_acc.append(history.history['acc'])
    with open(file, 'w+') as f:
        f.write(json.dumps(history_tr_acc))

    file = '../models/' + model_name + 'val_acc.json'
    if os.path.isfile(file):
        with open(file, 'r') as f:
            history_val_acc = json.load(f)
    else:
        history_val_acc = []
    history_val_acc.append(history.history['val_acc'])
    with open(file, 'w+') as f:
        f.write(json.dumps(history_val_acc))

    file = '../models/' + model_name + 'tr_loss.json'
    if os.path.isfile(file):
        with open(file, 'r') as f:
            history_tr_loss = json.load(f)
    else:
        history_tr_loss = []
    history_tr_loss.append(history.history['loss'])
    with open(file, 'w+') as f:
        f.write(json.dumps(history_tr_loss))

    file = '../models/' + model_name + 'val_loss.json'
    if os.path.isfile(file):
        with open(file, 'r') as f:
            history_val_loss = json.load(f)
    else:
        history_val_loss = []
    history_val_loss.append(history.history['val_loss'])
    with open(file, 'w+') as f:
        f.write(json.dumps(history_val_loss))

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


# 18 заебись

