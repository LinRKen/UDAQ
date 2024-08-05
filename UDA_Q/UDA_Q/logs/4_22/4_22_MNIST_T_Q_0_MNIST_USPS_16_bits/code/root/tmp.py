import torch
import logging
from tool import Intra_Norm

import matplotlib.pyplot as plt
import numpy as np
import os


addr = "/mnt/hdd1/zhangzhibin/dataset/USPS/22.jpg"

import torchvision.transforms as T
from datasets.make_dataloader import train_collate_fn

val_transforms = T.Compose([
    T.Resize((224, 224)),
    # T.Resize((256, 256)),
    # T.CenterCrop((224, 224)),
])

from PIL import Image
img = Image.open( addr ).convert('RGB')
img = val_transforms(img)
img.save('test_org.png')