import torch
import logging
from tool import Intra_Norm

import matplotlib.pyplot as plt
import numpy as np
import os

from base_models import model_vgg,model_vgg_bn
model = model_vgg_bn()
print(model)