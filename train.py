import prepare_dataset
from keras import models
from keras import layers

clean_path = 'D:\\nir_datasets\\jpg\\clean\\clean_cut_images1'
affected_path = 'D:\\nir_datasets\\jpg\\affected\\sorted\\vsl_lsb_5kb\\2019-03-17-15-04-40-207\\input000';
dataset_limit = 15000

images, labels = prepare_dataset.get_dataset(clean_path, affected_path, dataset_limit)
images_train, images_test, labels_train, labels_test = prepare_dataset.split_dataset(images, labels)

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(256 * 256,)))
network.add(layers.Dense(2, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

print('Fitting...')
print(images_train.shape)
print(labels_train.shape)
network.fit(images_train, labels_train, epochs=5, batch_size=128)

print('Testing...')
test_loss, test_acc = network.evaluate(images_test, labels_test)
print('Test loss: ', test_loss)
print('Test accuracy: ', test_acc)
