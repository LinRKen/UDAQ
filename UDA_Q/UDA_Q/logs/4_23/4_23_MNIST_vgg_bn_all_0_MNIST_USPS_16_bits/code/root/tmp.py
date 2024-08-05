import torch
import logging
from tool import Intra_Norm

import matplotlib.pyplot as plt
import numpy as np
import os

from base_models import model_vgg,model_vgg_bn_all
model = model_vgg_bn_all(32,10)
a = model.features.parameters()
for x in a:
    x.requires_grad_(False)