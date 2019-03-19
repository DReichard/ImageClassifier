
import os
import image_parsing
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

random_seed = 42


def randomize(dataset, labels):
    permutation = np.random.permutation(dataset.shape[0])
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def import_folder(clean_folder, limit):
    file_list = os.listdir(clean_folder)
    dataset = []
    for count, file in enumerate(file_list):
        if count >= limit:
            break
        if (count % 1000 == 0) & (count > 0):
            print("Files processed: " + str(count))
        image = image_parsing.get_image_grescale(os.path.join(clean_folder, file))
        dataset.append(image)
    print("Files processed total: " + str(count))
    dataset = np.array(dataset)
    return dataset


def get_dataset(clean_path, affected_path, limit):
    clean_images = import_folder(clean_path, limit)
    # print(clean_images.shape)
    clean_labels = np.full(clean_images.shape[0], 0, dtype=int)
    affected_images = import_folder(affected_path, limit)
    # print(affected_images.shape)
    affected_labels = np.full(affected_images.shape[0], 1, dtype=int)
    images = np.concatenate((clean_images, affected_images))
    # print("Dataset shape: " + str(images.shape))
    labels = np.concatenate((clean_labels, affected_labels))
    # print("Labels shape: " + str(labels.shape))
    # images, labels = randomize(images, labels)
    return images, labels


def split_dataset(images, labels):
    images_train, images_test, labels_train, labels_test = \
        train_test_split(images, labels, test_size=0.33, random_state=random_seed)

    images_train = images_train.reshape((images_train.shape[0], images_train.shape[1], images_train.shape[2], 1))
    images_train = images_train.astype('float32') / 255
    images_test = images_test.reshape((images_test.shape[0], images_test.shape[1], images_test.shape[2], 1))
    images_test = images_test.astype('float32') / 255
    labels_train = to_categorical(labels_train)
    labels_test = to_categorical(labels_test)
    return images_train, images_test, labels_train, labels_test
