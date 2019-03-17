import shutil
import os
import image_parsing


def cut_images(input_folder, output_folder):
    try:
        shutil.rmtree(output_folder)
    except FileNotFoundError:
        print('Cleaning failed, output path does not exist')
    os.mkdir(output_folder)
    file_list = os.listdir(input_folder)
    for count, file in enumerate(file_list):
        if file.endswith(".jpg") | file.endswith(".jpeg"):
            blocks = image_parsing.split_image_to_blocks(os.path.join(input_folder, file))
            for index, block in enumerate(blocks):
                output_path = os.path.join(output_folder, file.split('.')[0] + '_' + str(index) + '.jpg')
                image_parsing.save_array_as_greyscale_img(block, output_path)
        if count % 5000 == 0:
            print("Files processed: " + str(count))


print('Image cutting started')
cut_images('D:\\nir_datasets\\jpg\\clean_raw_images_input2', 'D:\\nir_datasets\\jpg\\clean\\clean_cut_images2')
print('Done')
