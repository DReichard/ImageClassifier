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
    return (channel.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


# TODO try not to butcher the lower-right corner mercilessly
def round_down(num, divisor):
    return num - (num % divisor)


def split_image_to_blocks(image_path):
    image = get_image(image_path)[:, :, 0]  # TODO RED CHANNEL ONLY
    image_height, image_width = image.shape
    image = image[:round_down(image_height, 256), :round_down(image_width, 256)]
    if any(dimension == 0 for dimension in image.shape):
        return np.array([])
    image_blocks = split_blocks(image, 256, 256)
    return image_blocks


# image = get_image('D:\\nir_datasets\\jpg\\vsl_in\\Trithemiusmoredetail.jpg')[:, :, 0]
