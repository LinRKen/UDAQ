import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import pickle
import scipy.io
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# config = tf.compat.v1.ConfigProto()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9

# Dataset path
# Source: 5,000, Target: 54,000
# Gallery: 54,000 Query: 1,000
# data_dir = './cifar10'
data_dir = '/mnt/hdd1/zhangzhibin/dataset/CIFAR-10/cifar-10-batches-py/'


# For training
ImagNet_pretrained_path = './models/ImageNet_pretrained'
# model_save_path = './models/OfficeHome_4/'

# For evaluation
# model_load_path = './models/48bits_example.ckpt'
# model_load_path = './models/OfficeHome/R_A/100.ckpt'
# /home/zhangzhibin/data/UDA/CDTrans-master/GPQ/models/.meta
# cifar10_label_sim_path = './cifar10/cifar10_Similarity.mat'
# cifar10_label_sim_path = '/mnt/hdd1/zhangzhibin/dataset/CIFAR-10/cifar-10-batches-py/cifar10_Similarity.mat'
# /mnt/hdd1/zhangzhibin/dataset/CIFAR-10/cifar-10-batches-py/


n_CLASSES = 10
image_size = 32
img_channels = 3
n_DB = 70000

# n_CLASSES = 65
# image_size = 32
# img_channels = 3
# n_DB = 4357

# n_CLASSES = 345
# image_size = 32
# img_channels = 3
# n_DB = 120906

'Hyperparameters for training'
# Training epochs, 1 epoch represents training all the source data once
# total_epochs = 20
batch_size = 500
# save model for every save_term-th epoch
save_term = 5

# length of codeword
# len_code = 6
len_code = 12
print('len_code = ',len_code)

# Number of codebooks
# n_book = 12
# n_book = 8
# n_book = 4

# Number of codewords=(2^bn_word)
# bn_word = 4
# bn_word = 8


# Number of bits for retrieval
# n_bits = n_book * bn_word

# Soft assignment input scaling factor
alpha = 20.0

# Classification input scaling factor
beta = 4

# lam1, 2: loss function balancing parameters
lam_1 = 0.1
lam_2 = 0.1


