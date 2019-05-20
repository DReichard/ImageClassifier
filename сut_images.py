import shutil
import os
import image_parsing
from utility_scripts import pgm_io


def cut_images_jpg(input_folder, output_folder, size):
    try:
        shutil.rmtree(output_folder)
    except FileNotFoundError:
        print('Cleaning failed, output path does not exist')
    os.mkdir(output_folder)
    file_list = os.listdir(input_folder)
    for count, file in enumerate(file_list):
        if file.endswith(".jpg") | file.endswith(".jpeg")| file.endswith('.png')| file.endswith('.pgm'):
            blocks = image_parsing.split_image_to_blocks(os.path.join(input_folder, file), size)
            for index, block in enumerate(blocks):
                output_path = os.path.join(output_folder, file.split('.')[0] + '_' + str(index) + '.png')
                image_parsing.save_array_as_greyscale_img(block, output_path)
        if count % 100 == 0:
            print("Files processed: " + str(count / len(file_list)))


def cut_images_bgm(input_folder, output_folder, size):
    try:
        shutil.rmtree(output_folder)
    except FileNotFoundError:
        print('Cleaning failed, output path does not exist')
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder, "test"))
    os.mkdir(os.path.join(output_folder, "train"))
    file_list = os.listdir(input_folder)
    idx = 0
    dest = "test"
    for count, file in enumerate(file_list):
        if file.endswith(".jpg") | file.endswith(".jpeg")| file.endswith('.png'):
            blocks = image_parsing.split_image_to_blocks(os.path.join(input_folder, file), size)
            for index, block in enumerate(blocks):
                if idx >= 1000:
                    dest = "train"
                output_path = os.path.join(output_folder, dest, file.split('.')[0] + '_' + str(index) + '.pgm')
                pgm_io.pgmwrite(block, output_path)
                idx = idx + 1
        if count % 20 == 0:
            print("Files processed: " + str(int(count / len(file_list) * 100)) + '%')


print('Image cutting started')
# cut_images_bgm('D:\\nir_datasets\\jpg\\my_memes_raw', 'D:\\nir_datasets\\png\\clean\\128', 128)
cut_images_jpg('C:\\datasets\\s_uniward_128\\test\\stego', 'D:\\nir_datasets\\png\\s-uniward\\128', 128)
print('Done')
