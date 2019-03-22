from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import prepare_dataset
import rdn_assembly
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu',
#                         input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# model.summary()
#model.compile(optimizer='rmsprop',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
print('Network Compiled')



clean_path = 'D:\\nir_datasets\\png\\clean\\clean_cut_stl10'
affected_path = 'D:\\nir_datasets\\png\\affected\\stl-10\\2019-03-20-23-10-51-808\\input000'
dataset_limit = 1000

images, labels = prepare_dataset.get_dataset(clean_path, affected_path, dataset_limit)

#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#images = np.concatenate((train_images, test_images))
#labels = np.concatenate((train_labels, test_labels))
train_images, test_images, train_labels, test_labels = prepare_dataset.split_dataset(images, labels)

# odel.compile(optimizer='rmsprop',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

model = rdn_assembly.get_network()

model.fit(train_images, train_labels, batch_size=50, epochs=500, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=40)
print(test_acc)
