
from PIL import Image
from keras.engine.saving import load_model
import matplotlib.pyplot as plt

from assembling_scripts.cnn_xu_assembly import assemble_network_xu
from assembling_scripts.rdn_assembly import get_network
from training_scripts.batch_generator import pair_generator
from keras.utils import plot_model

train_cover_path = r"C:\datasets\j-uniward_256_2020\train\cover"
train_stego_path = r"C:\datasets\j-uniward_256_2020\train\stego"
test_cover_path = r"C:\datasets\j-uniward_256_2020\test\cover"
test_stego_path = r"C:\datasets\j-uniward_256_2020\test\stego"

image_size = 128
batch_size = 4
do_shuffle = False
train_set_size = 4
test_set_size = 2

train_epoch_size = train_set_size / batch_size
# train_epoch_size = 2 / batch_size
test_epoch_size = test_set_size / batch_size
# test_epoch_size = 2 / batch_size

train_generator = pair_generator(train_cover_path, train_stego_path, image_size, batch_size, train_set_size, do_shuffle)
test_generator = pair_generator(test_cover_path, test_stego_path, image_size, batch_size, test_set_size, do_shuffle)

network = get_network()
ep = 1

i = 0

# plot_model(network, to_file='model.png')


while True:
    # network = load_model('../models/model_rdn_' + str(i) + 'full.h5')
    LAYER_NAME = 'conv2d_1'
    # act_map = visualize_activation(network, 1, filter_indices=None, seed_input=None, \
    #                                input_range=(0, 255), backprop_modifier=None, grad_modifier=None, act_max_weight=1, \
    #                                lp_norm_weight=10, tv_weight=10)[0, :, :]
    # img = Image.fromarray(act_map)
    # plt.imshow(img)
    i += ep
    print('Epoch: ' + str(i))
    history = network.fit_generator(train_generator, epochs=ep, steps_per_epoch=train_epoch_size, shuffle=False, validation_data=test_generator, validation_steps=test_epoch_size, verbose=1)

    open("../models/model_rdn_"+str(i)+".json", 'w').write(network.to_json())
    network.save_weights("../models/model_rdn_"+str(i)+".h5")
    network.save("../models/model_rdn_"+str(i)+"full.h5")
    # onnx_model = onnxmltools.convert_keras(network, target_opset=7)
    # onnxmltools.utils.save_text(onnx_model, '../models/model_xu_' + str(i) + '_onnx.json')
    # onnxmltools.utils.save_model(onnx_model, '../models/model_xu_' + str(i) + '_onnx.onnx')

