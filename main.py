from PIL import Image
import numpy as np


def get_image(image_path):
    img = Image.open(image_path)
    image_array = np.array(img)
    return image_array


def split_blocks(channel, nrows, ncols):
    h, w = channel.shape
    return (channel.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


# image = get_image('D:\\nir_datasets\\jpg\\vsl_in\\CNcRzAPx9a4.jpg')
# make it for in
image = get_image('D:\\nir_datasets\\jpg\\vsl_in\\CNcRzAPx9a4.jpg')[0]

# image_blocks = split_blocks(image, 256, 256);
# make it for in
image_blocks = split_blocks(image, 256, 256)[0];


