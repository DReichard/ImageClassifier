from keras.models import load_model
import onnxmltools
import tensorflowjs as tfjs
import tensorflow as tf


tf.compat.v1.disable_eager_execution()
input = r"C:\Users\stani\PycharmProjects\ImageClassifier\models\model_xu_2020_j_float32_undiv_1full.h5"

output_onnx = input.replace(".h5", ".onnx")
output_json = input.replace(".h5", ".json")
print("loading model")
network = load_model(input)
print("converting model")
onnx_model = onnxmltools.convert_keras(network, target_opset=7)
print("serializing model")
onnxmltools.utils.save_text(onnx_model, output_json)
onnxmltools.utils.save_model(onnx_model, output_onnx)
tfjs.converters.save_keras_model(network, r"C:\Users\stani\PycharmProjects\ImageClassifier\models")

print("done")
