import torch
import logging
from tool import Intra_Norm

import matplotlib.pyplot as plt
import numpy as np
import os


predict_prob = torch.ones(10)
predict_prob.requires_grad_(True)
# soft_weight = torch.zeros_like( predict_prob )

# soft_weight.data = torch.sigmoid( 10*(predict_prob.detach()-0.5) )
soft_weight = torch.sigmoid( 10*(predict_prob.detach()-0.5) )

loss = torch.sum( soft_weight * predict_prob)
print(loss)
loss.backward()
print(predict_prob.grad)
print(soft_weight.grad)