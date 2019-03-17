import prepare_dataset
from keras import models
from keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

clean_path = 'D:\\nir_datasets\\jpg\\clean\\clean_cut_images1'
affected_path = 'D:\\nir_datasets\\jpg\\affected\\sorted\\vsl_lsb_5kb\\2019-03-17-15-04-40-207\\input000';
dataset_limit = 5000
random_seed = 42

images, labels = prepare_dataset.get_dataset(clean_path, affected_path, dataset_limit)

images_train, images_test, labels_train, labels_test = \
    train_test_split(images, labels, test_size=0.33, random_state=random_seed)

images_train = images_train.reshape((images_train.shape[0], images_train.shape[1] * images_train.shape[2]))
images_test = images_test.reshape((images_test.shape[0], images_test.shape[1] * images_test.shape[2]))

labels_train = to_categorical(labels_train)
labels_test = to_categorical(labels_test)

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
