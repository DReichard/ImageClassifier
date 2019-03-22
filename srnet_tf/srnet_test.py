import os
import sys
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # set a GPU (with GPU Number)
home = os.path.expanduser("~")
sys.path.append(home + '/tflib/')        # path for 'tflib' folder

from srnet_tf.SRNet import *

train_batch_size = 32
valid_batch_size = 40
max_iter = 500000
train_interval = 100
valid_interval = 5000
save_interval = 5000
num_runner_threads = 10

# Cover and Stego directories for training and validation. For the spatial domain put cover and stego images in their
# corresponding direcotries. For the JPEG domain, decompress images to the spatial domain without rounding to integers and
# save them as '.mat' files with variable name "im". Put the '.mat' files in thier corresponding directoroies. Make sure
# all mat files in the directories can be loaded in Python without any errors.

TRAIN_COVER_DIR = 'D:\\nir_datasets\\jpg\\clean\\1000_clean_cut_'
TRAIN_STEGO_DIR = 'D:\\nir_datasets\\jpg\\affected\\sorted\\1000_lsb'

VALID_COVER_DIR = 'D:\\nir_datasets\\jpg\\clean\\100_clean'
VALID_STEGO_DIR = 'D:\\nir_datasets\\jpg\\affected\\sorted\\100_lsb'

train_gen = partial(gen_flip_and_rot, TRAIN_COVER_DIR, TRAIN_STEGO_DIR)
valid_gen = partial(gen_valid, VALID_COVER_DIR, VALID_STEGO_DIR)

LOG_DIR = '/media/LogFiles/JUNI_75_04/'  # path for a log direcotry
# load_path = LOG_DIR + 'Model_460000.ckpt'  # continue training from a specific checkpoint
load_path = None  # training from scratch

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

train_ds_size = len(glob(TRAIN_COVER_DIR + '/*')) * 2
valid_ds_size = len(glob(VALID_COVER_DIR + '/*')) * 2
print('train_ds_size: ', train_ds_size)
print('valid_ds_size: ', valid_ds_size)

if valid_ds_size % valid_batch_size != 0:
    raise ValueError("change batch size for validation")

optimizer = AdamaxOptimizer
boundaries = [400000]  # learning rate adjustment at iteration 400K
values = [0.001, 0.0001]  # learning rates

train(SRNet, train_gen, valid_gen, train_batch_size, valid_batch_size, valid_ds_size, optimizer, boundaries, values, train_interval, valid_interval, max_iter, save_interval, LOG_DIR, num_runner_threads, load_path)
# Testing
# Cover and Stego directories for testing
TEST_COVER_DIR = '/media/TST/Cover/'
TEST_STEGO_DIR = '/media/TST/JUNI_75_04/'

test_batch_size=40
LOG_DIR = '/media/LogFiles/JUNI_75_04/'
LOAD_CKPT = LOG_DIR + 'Model_435000.ckpt'        # loading from a specific checkpoint

test_gen = partial(gen_valid, TEST_COVER_DIR, TEST_STEGO_DIR)

test_ds_size = len(glob(TEST_COVER_DIR + '/*')) * 2
print('test_ds_size: ', test_ds_size)

if test_ds_size % test_batch_size != 0:
    raise ValueError("change batch size for testing!")

test_dataset(SRNet, test_gen, test_batch_size, test_ds_size, LOAD_CKPT)
