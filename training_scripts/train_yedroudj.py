import onnxmltools

from assembling_scripts.cnn_yedroudj_assembly import assemble_network_tedroudj
from training_scripts.batch_generator import pair_generator


def magic_spell():
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


magic_spell()

train_cover_path = "C:\\datasets\\f5_128\\train\\cover"
train_stego_path = "C:\\datasets\\f5_128\\train\\stego"

test_cover_path = "C:\\datasets\\f5_128\\test\\cover"
test_stego_path = "C:\\datasets\\f5_128\\test\\stego"

image_size = 128
batch_size = 2
train_epoch_size = 5250
test_epoch_size = 91


train_generator = pair_generator(train_cover_path, train_stego_path, image_size, batch_size)
test_generator = pair_generator(test_cover_path, test_stego_path, image_size, 1)

network = assemble_network_tedroudj(image_size)

ep = 1
i = 0
while True:
    i += ep
    network.fit_generator(train_generator, epochs=ep, steps_per_epoch=train_epoch_size, validation_data=test_generator, validation_steps=test_epoch_size, verbose=1)
    open("../models/yedroudj/model_yedroudj_"+str(i)+".json", 'w').write(network.to_json())
    network.save_weights("../models/yedroudj/model_yedroudj_"+str(i)+".h5")
    network.save("../models/yedroudj/model_yedroudj_"+str(i)+"full.h5")
    onnx_model = onnxmltools.convert_keras(network, target_opset=7)
    onnxmltools.utils.save_text(onnx_model, '../models/yedroudj/model_yedroudj_' + str(i) + '_onnx.json')
    onnxmltools.utils.save_model(onnx_model, '../models/yedroudj/model_yedroudj_' + str(i) + '_onnx.onnx')


# network.evaluate_generator(test_generator)
