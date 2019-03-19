from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

import prepare_dataset
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('Network Compiled')

clean_path = 'D:\\nir_datasets\\jpg\\clean\\clean_cut_images1'
affected_path = 'D:\\nir_datasets\\jpg\\affected\\sorted\\vsl_lsb_5kb\\2019-03-17-15-04-40-207\\input000';
dataset_limit = 1000

# images, labels = prepare_dataset.get_dataset(clean_path, affected_path, dataset_limit)


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
images = np.concatenate((train_images, test_images))
labels = np.concatenate((train_labels, test_labels))
train_images, test_images, train_labels, test_labels = prepare_dataset.split_dataset(images, labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
