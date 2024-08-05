from cmath import tanh
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from torch.utils.data.dataloader import DataLoader
from base_Q import Base_Q

import time
import torch.optim as optim
from tqdm import tqdm
from icecream import ic

class MSE_PQ(nn.Module):
    def __init__(self,n_quantizer,n_codeword, len_vec,lr=1e-3):
        super(MSE_PQ,self).__init__()
        self.n_quantizer = n_quantizer
        self.n_codeword = n_codeword
        self.len_vec = len_vec
        self.len_subvec = len_vec // n_quantizer
        assert( self.len_subvec * n_quantizer == len_vec )
        
        self.list_Q = [ Base_Q(n_codeword,self.len_subvec).cuda() for _ in range(n_quantizer) ]
        
        parameters = [ self.list_Q[i].CodeBook for i in range(n_quantizer) ]
        # self.optimizer = optim.SGD( parameters, momentum=0.9, nesterov=True, lr=lr)
        self.optimizer = optim.Adam( parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    
    def forward(self,x,soft_rate=-1):
        x = x.view( x.size(0) , self.n_quantizer  , self.len_subvec )
        Q_x = torch.zeros_like( x )
        id_x = torch.LongTensor( self.n_quantizer, x.size(0)).cuda()
        for deep in range(self.n_quantizer):
            Q_x[:,deep,:],id_x[deep] = self.list_Q[deep]( x[:,deep,:] , soft_rate )
        return Q_x.view(-1,self.len_vec),id_x
        

    def calc_loss(self,x,soft_rate=0.1,hard_rate=10):
        Q_soft,_ = self.forward( x , soft_rate )
        Q_hard,_ = self.forward( x , hard_rate )
        Q_soft_error = torch.norm( Q_soft - x , p=2 , dim=1 ).mean()
        Q_hard_error = torch.norm( Q_hard - x , p=2 , dim=1 ).mean()
        return Q_soft_error , Q_hard_error

    def div_loss(self):
        div_loss = 0
        for Q in self.list_Q:
            dist = torch.cdist( Q.CodeBook , Q.CodeBook ,p=2) + torch.eye( self.n_codeword).cuda()*1e6
            
            MIN_id = torch.argmin(dist, 1)
            MIN_dist = torch.gather( dist, 1, MIN_id.view(-1,1))
            div_loss += MIN_dist.min()
            
        div_loss /= self.n_quantizer
        return -torch.tanh( div_loss )
    
    def lr_adjust(self):
        self.optimizer.param_groups[0]['lr'] = max( 5e-4 , self.optimizer.param_groups[0]['lr']*0.9 )
    
    def parameters(self):
        paras = []
        for base_Q in self.list_Q:
            paras.append( base_Q.CodeBook )
        return paras

    def train_epoch(self,epoch,loader_image,model=None,soft_rate=0.1,hard_rate = 10):
        st_time = time.time()
        mean_loss = 0
        mean_soft_loss = 0
        mean_hard_loss = 0
        total_data = 0
        # for i, data in tqdm(enumerate(loader_image), desc='Processing'):
        
        # div_flag = True
        div_flag = False
        if div_flag :
            print('========  train with div_loss  =====')
        output_batch = 1000
        for i, data in enumerate(loader_image):
            im = data[0]
            self.optimizer.zero_grad()

            im = im.float().cuda()

            if model is not None:
                with torch.no_grad():
                    _, im_feat = model(im)
                    im_em = model.base_model.last_linear(im_feat)
            else:
                im_em = im
            total_data += im_em.size(0)

            loss_soft , loss_hard = self.calc_loss(im_em,soft_rate=soft_rate,hard_rate=hard_rate)
            # loss = loss_hard + 0.1*loss_soft

            loss = loss_hard + loss_soft

            if div_flag :
                loss += 0.1*self.div_loss()
            # loss += self.div_loss()
            if ( i+1)%output_batch==0:
                print('train OK ',i+1)

            loss.backward()
            self.optimizer.step()
            
            mean_loss += loss.item()
            mean_soft_loss+= loss_soft.item()
            mean_hard_loss+= loss_hard.item()
        mean_loss /= total_data
        mean_soft_loss /= total_data
        mean_hard_loss /= total_data

        self.lr_adjust()
        print(f'epoch = {epoch:d} loss = {mean_loss:.3f} soft_loss = {mean_soft_loss:.3f} hard_loss = {mean_hard_loss:.3f}\n')
