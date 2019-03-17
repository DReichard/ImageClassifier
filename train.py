from keras import models
from keras import layers
import prepare_dataset
import rdn_assembly

clean_path = 'D:\\nir_datasets\\jpg\\clean\\clean_cut_images1'
affected_path = 'D:\\nir_datasets\\jpg\\affected\\sorted\\vsl_lsb_5kb\\2019-03-17-15-04-40-207\\input000';
dataset_limit = 3000

images, labels = prepare_dataset.get_dataset(clean_path, affected_path, dataset_limit)
images_train, images_test, labels_train, labels_test = prepare_dataset.split_dataset(images, labels)

network = rdn_assembly.get_network()

print('Fitting...')
network.fit(images_train, labels_train, epochs=5, batch_size=1500)

print('Testing...')
# test_loss, test_acc = network.evaluate(images_test, labels_test)
# print('Test loss: ', test_loss)
# print('Test accuracy: ', test_acc)
