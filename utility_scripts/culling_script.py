import os


def diff(li1, li2):
    return (list(set(li1) - set(li2)))


def cull_directories(cover_root, stego_root):
    cover_paths = set(os.listdir(cover_root))
    stego_paths = set(os.listdir(stego_root))
    cover_to_trim = diff(cover_paths, stego_paths)
    stego_to_trim = diff(stego_paths, cover_paths)

    for the_file in cover_to_trim:
        file_path = os.path.join(cover_root, the_file)
        try:
            if os.path.isfile(file_path):
                print("deleting: "+ file_path)
                os.unlink(file_path)
        except Exception as e:
            print(e)

    for the_file in stego_to_trim:
        file_path = os.path.join(stego_root, the_file)
        try:
            if os.path.isfile(file_path):
                print("deleting: "+ file_path)
                os.unlink(file_path)
        except Exception as e:
            print(e)


train_cover_path = r"C:\datasets\j-uniward_128_2020\cover"
train_stego_path = r"C:\datasets\j-uniward_128_2020\stego"

cull_directories(train_cover_path, train_stego_path)
