from PIL import Image
import numpy as np


def save_array_as_greyscale_img(array, output_path):
    img = Image.fromarray(array)
    img.save(output_path)


def get_image(image_path):
    img = Image.open(image_path)
    image_array = np.array(img)
    return image_array


def get_image_grescale(image_path):
    img = Image.open(image_path)
    image_array = np.array(img)
    return image_array[:, :]


def split_blocks(channel, nrows, ncols):
    h, w = channel.shape
    try:
        res = (channel.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))
    except ValueError:
        res = ()
    return res


# TODO try not to butcher the lower-right corner mercilessly
def round_down(num, divisor):
    return num - (num % divisor)


def split_image_to_blocks(image_path, size):
    try:
        image = get_image(image_path)[:, :, 0]  # TODO RED CHANNEL ONLY
    except IndexError:
        image = get_image(image_path)[:, :]
    image_height, image_width = image.shape
    if image_height > size:
        image = image[:round_down(image_height, size), :round_down(image_width, size)]
    if any(dimension == 0 for dimension in image.shape):
        return np.array([])
    image_blocks = split_blocks(image, size, size)
    return image_blocks

