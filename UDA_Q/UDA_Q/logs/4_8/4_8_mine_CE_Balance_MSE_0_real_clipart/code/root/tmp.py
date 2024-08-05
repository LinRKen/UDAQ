import torch
import logging
from tool import Intra_Norm

import matplotlib.pyplot as plt
import numpy as np


def draw_3D(data,name):
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111, projection='3d')
    # fake data
    _x = np.arange(data.shape[0])
    _y = np.arange(data.shape[1])
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = data.ravel()
    bottom = np.zeros_like(top)
    width = depth = 0.35

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    plt.savefig(name)

data = np.array( [ [0,1,2],[0,2,0],[3,2,5] ])

draw_3D(data,'tmp.png')

# logger = logging.getLogger('UDA_Q log')
# logger.setLevel(level = logging.INFO)

# file_handler = logging.FileHandler("log.txt",mode='w')
# formatter = logging.Formatter('%(asctime)s - %(message)s',"%Y-%m-%d-%H:%M:%S")
# file_handler.setFormatter(formatter)

# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(formatter)

# logger.addHandler(file_handler)
# logger.addHandler(console_handler)

# a=0
# logger.info(f"Start print log {a:d}")
# logger.info(f"Start print log {a:d}")
