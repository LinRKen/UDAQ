import torch
import logging
from tool import Intra_Norm

import matplotlib.pyplot as plt
import numpy as np
import os

from base_models import model_vgg,model_vgg_bn_all
from my_dataset import get_img_MNIST_USPS
from fixed_dataset import get_MNIST_USPS

S_train_set, S_query_set, S_gallery_set, T_train_set, T_query_set, T_gallery_set = get_MNIST_USPS()

cls = [0]*10
for _,label in S_query_set:
    cls[label] +=1
print(cls)
