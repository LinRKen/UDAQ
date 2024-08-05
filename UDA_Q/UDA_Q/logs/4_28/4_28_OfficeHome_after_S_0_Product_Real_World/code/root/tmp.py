import torch
import logging
from tool import Intra_Norm

import matplotlib.pyplot as plt
import numpy as np
import os

from base_models import model_vgg,model_vgg_bn_all
from my_dataset import get_img_MNIST_USPS
from fixed_dataset import get_MNIST_USPS

a = model_vgg()
print(a)