from keras import models
from keras import layers
import numpy as np
import image_parsing
import prepare_dataset
import rdn_assembly
from PIL import Image

clean_path = 'D:\\nir_datasets\\jpg\\clean\\clean_cut_images1'
affected_path = 'D:\\nir_datasets\\jpg\\affected\\sorted\\vsl_lsb_5kb\\2019-03-17-15-04-40-207\\input000';
dataset_limit = 2000

images, labels = prepare_dataset.get_dataset(clean_path, affected_path, dataset_limit)
images = images.astype('float32')/255
images_train, images_test, labels_train, labels_test = prepare_dataset.split_dataset(images, labels)

network = rdn_assembly.get_network_test()
print('Fitting...')
network.fit(images_train, labels_train, epochs=5, batch_size=30)

print('Testing...')
# test_loss, test_acc = network.evaluate(images_test, labels_test)
# print('Test loss: ', test_loss)
# print('Test accuracy: ', test_acc)
