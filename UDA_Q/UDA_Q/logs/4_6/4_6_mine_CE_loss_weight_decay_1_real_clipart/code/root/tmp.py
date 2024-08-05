import torch
import logging
from tool import Intra_Norm

a = torch.FloatTensor( (1,2,2,1,1,1) ).view(1,-1)
a = Intra_Norm(a,3)
print(a)

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
