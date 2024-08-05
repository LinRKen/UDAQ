import torch
import logging
from tool import Intra_Norm

import matplotlib.pyplot as plt
import numpy as np
import os

from base_models import model_vgg,model_vgg_bn_all
from my_dataset import get_img_MNIST_USPS

a = get_img_MNIST_USPS()[3]
pic = a[0][0]
print(pic.size())
print(pic.mean())
for i in range(pic.size(1)):
    for j in range(pic.size(2)):
        if ( pic[0][i][j]>0):
            print(f'{pic[0][i][j]:.3f}',end=' ')
    print('')