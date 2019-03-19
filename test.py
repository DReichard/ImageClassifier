from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical

import prepare_dataset

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(256, 256, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('Network Compiled')

clean_path = 'D:\\nir_datasets\\jpg\\clean\\clean_cut_images1'
affected_path = 'D:\\nir_datasets\\jpg\\affected\\sorted\\vsl_lsb_5kb\\2019-03-17-15-04-40-207\\input000';
dataset_limit = 250

images, labels = prepare_dataset.get_dataset(clean_path, affected_path, dataset_limit)

train_images, test_images, train_labels, test_labels = prepare_dataset.split_dataset2(images, labels)

#train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
#train_images = train_images.astype('float32') / 255
#test_images = test_images.reshape((10000, 28, 28, 1))
#test_images = test_images.astype('float32') / 255
#train_labels = to_categorical(train_labels)
#test_labels = to_categorical(test_labels)


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=20, batch_size=16)

