import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.utils.data import Dataset
from icecream import ic

class Base_Q(nn.Module):
    def __init__(self, n_codeword, len_vec):
        super(Base_Q, self).__init__()
        self.n_codeword = n_codeword
        self.len_vec = len_vec
        self.CodeBook = torch.nn.Parameter(torch.rand(n_codeword, len_vec))
        torch.nn.init.normal_(self.CodeBook.data, std= 0.2)

    def forward(self,x,soft_rate=-1):
        dist = -torch.cdist(x, self.CodeBook , p=2 )
        if soft_rate >0 :
            soft_dist = F.softmax( dist*soft_rate , dim =1 )
            Q_soft = torch.mm( soft_dist , self.CodeBook )
            return Q_soft , -1
        else:
            MAX_id = torch.argmax(dist, 1)
            Q_hard = torch.index_select(self.CodeBook, 0, MAX_id)
            return Q_hard , MAX_id

class Dataset_fixed_f(Dataset):
    def __init__(self,data) -> None:
        self.data = data
    def __getitem__(self,id):
        return torch.from_numpy(self.data[id]) , 0
    def __len__(self):
        return self.data.shape[0]