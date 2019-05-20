import multiprocessing
import os
import random
import shutil
import string
import numpy
import subprocess


def random_string(string_length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))


def run_win_cmd(cmd):
    pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        line = pipe.stdout.readline()
        error = pipe.stderr.readline()
        if error:
            return 1
        if line:
            print
            line
        if not line:
            break
        return 0


def embed_f5(input_folder, clean_output_folder, stego_output_folder, tmp_folder, payload_length):
    file_list = os.listdir(input_folder)
    print("F5 Embedding with " + str(payload_length) + " bytes started")
    fail_count = 0
    for count, cover_file in enumerate(file_list):
        if cover_file.endswith(".jpg") | cover_file.endswith(".jpeg") | cover_file.endswith('.png'):
            payload_tmp_file_path = os.path.join(tmp_folder, str(count) + '.txt')
            stego_file_path = os.path.join(stego_output_folder, cover_file)
            numpy.savetxt(payload_tmp_file_path, [random_string(payload_length)], fmt='%s')
            cover_file_full_path = os.path.join(input_folder, cover_file)
            cmd_arr = ["f5stego", "-e", "-p", random_string(8), cover_file_full_path, payload_tmp_file_path, stego_file_path]
            cmd = ' '.join(map(str, cmd_arr))

            code = run_win_cmd(cmd)
            if code == 0:
                out_clean_path = os.path.join(clean_output_folder, cover_file)
                shutil.copy(cover_file_full_path, out_clean_path)
            else:
                fail_count = fail_count + 1
                if fail_count % 10 == 0:
                    print("Fail count: " + str(fail_count))
            if count % 50 == 0:
                print(str(count) + " files processed")
            os.remove(payload_tmp_file_path)
    print("F5 Embedding with " + str(payload_length) + " bytes done")


source = "D:\\nir_datasets\\jpg\\clean\\clean_128"
tmp_path = "D:\\nir_datasets\\tmp"
out_stego_path = "C:\\datasets\\f5_128\\test2\\stego"
out_cover_path = "C:\\datasets\\f5_128\\test2\\cover"
payload_length = 128

embed_f5(source, out_cover_path, out_stego_path, tmp_path, payload_length)

