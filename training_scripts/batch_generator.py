import os
import random
import threading
from itertools import chain, islice

import numpy as np
from keras.utils import np_utils
from scipy import misc, ndimage, signal
from skimage.util.shape import view_as_blocks, view_as_windows

from utility_scripts import pgm_io


def load_image(path, image_size):
    # I = misc.imread(path)
    I = pgm_io.pgmread(path)
    # patches = view_as_blocks(I, (image_size, image_size))
    # return patches[0, 0]
    return I


def pair_generator_internal(cover_root, stego_root, image_size, set_size, do_shuffle):
    cover_paths = np.array(sorted(os.listdir(cover_root))[:set_size])
    stego_paths = np.array(sorted(os.listdir(stego_root))[:set_size])
    inner = np.array(list(set(cover_paths) & set(stego_paths)))
    indices = np.arange(inner.shape[0])
    print(inner[0] + "\n --- \n" + inner[-1])
    while True:
        if do_shuffle:
            np.random.shuffle(indices)

        inner = inner[indices]

        for name in inner:
            cover_path = os.path.join(cover_root, name)
            stego_path = os.path.join(stego_root, name)
            cover_image = load_image(cover_path, image_size)
            stego_image = load_image(stego_path, image_size)
            if cover_image is None:
                continue
            if stego_image is None:
                continue
            if stego_image.size == 0:
                continue
            try:
                if cover_image.shape[0] != image_size | cover_image.shape[1] != image_size | stego_image.shape[0] != image_size | stego_image.shape[1] != image_size:
                    continue
            except:
                continue
            x = (cover_image, stego_image)
            y = ([0], [1])
            # y = np_utils.to_categorical(y, 2)
            yield (x, y)


def pair_generator(cover_path, stego_path, image_size, batch_size, set_size, do_shuffle):
    iterable = pair_generator_internal(cover_path, stego_path, image_size, set_size, do_shuffle)
    iterator = iter(iterable)
    for first in iterator:
        records_batch = chain([first], islice(iterator, batch_size - 1))
        records_nested = [x for x in records_batch]
        images_nested = [l[0] for l in records_nested]
        labels_nested = [l[1] for l in records_nested]
        images_flat = np.array([e for l in images_nested for e in l])
        images_flat = np.expand_dims(images_flat, axis=1)
        labels_flat = np.array([e for l in labels_nested for e in l])
        labels_flat = np_utils.to_categorical(labels_flat, 2)
        yield (images_flat, labels_flat)


