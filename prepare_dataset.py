import shutil
import image_parsing
import os


def prepare_dataset(input_folder, output_folder):
    try:
        shutil.rmtree(output_folder)
    except FileNotFoundError:
        print('Cleaning failed, output path does not exist')
    os.mkdir(output_folder)
    file_list = os.listdir(input_folder)
    for file in file_list:
        if file.endswith(".jpg") | file.endswith(".jpeg"):
            blocks = image_parsing.split_image_to_blocks(os.path.join(input_folder, file))
            for index, block in enumerate(blocks):
                output_path = os.path.join(output_folder, file.split('.')[0] + '_' + str(index) + '.jpg')
                image_parsing.save_array_as_greyscale_img(block, output_path)


prepare_dataset('D:\\nir_datasets\\jpg\\input', 'D:\\nir_datasets\\jpg\\dataset')

