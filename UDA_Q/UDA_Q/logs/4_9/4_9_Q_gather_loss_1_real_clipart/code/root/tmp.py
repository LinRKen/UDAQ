import torch
import logging
from tool import Intra_Norm

import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

lis = [1,3,4,1,5,5,1,1,23,21,1,12,3]

high = 100000
b = torch.LongTensor( (3,1,2) )
a = torch.randint( 0, high , b.size())
print(a)
a = a%b
print(a)
print(lis[a])