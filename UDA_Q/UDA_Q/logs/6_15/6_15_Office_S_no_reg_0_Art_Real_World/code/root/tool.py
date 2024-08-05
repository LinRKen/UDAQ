import torch

class Avg_er():
    def __init__(self,name):
        self.sum = 0
        self.mean = 0
        self.cnt = 0
        self.name = name

    def add(self,n_mean,n_cnt):
        self.sum += n_mean*n_cnt
        self.cnt += n_cnt
        self.mean = self.sum/self.cnt
    
    def out_s(self):
        return self.name+' = '+f'{self.mean:.3f}'


def Intra_Norm(x,num_split=4):
    x = x.view( x.size(0),num_split,-1)
    x = torch.nn.functional.normalize(x,dim=2)
    return x.view(x.size(0),-1)