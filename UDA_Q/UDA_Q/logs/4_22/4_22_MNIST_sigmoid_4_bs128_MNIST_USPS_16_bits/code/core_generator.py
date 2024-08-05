from numpy.core.fromnumeric import argmax
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from sys import argv
import numpy as np

from torch.utils import data

def get_cos_dist(a,b):
    norm_1 = torch.norm(a,dim=1).repeat(b.size(0),1).t()

    norm_2 = torch.norm(b,dim=1).repeat(a.size(0),1)
    
    tmp = torch.mm(a,b.t())
    c = torch.mm(a,b.t())/norm_1/norm_2
    return c

def out_float_tensor(A):
    if ( A.dim() == 1 ):
        for i in range( A.size(0) ):
            print('%.3f'%(A[i]),end=' ')
        print('')
    else:
        assert(A.dim()==2)
        for i in range( A.size(0) ):
            for j in range( A.size(1)):
                print('%.3f'%(A[i][j]),end=' ')
            print('')

def generate_core(n_core,dim):
    n_epoch = 1000

    margin = torch.eye(n_core)*2-1

    core = torch.nn.Parameter ( torch.normal( mean = torch.zeros( n_core , dim ) , std = 2 ) )


    optimizer = optim.SGD( [core] , lr=1, momentum=0.9 , nesterov=True )
    scheduler = lr_scheduler.StepLR(optimizer,step_size=2000,gamma = 0.9 )

    epoch_iter = n_epoch//10

    for epoch in range( n_epoch ):

        optimizer.zero_grad()
        
        tmp_dist = get_cos_dist( core , core )
        # masked_dist_zero = torch.where( mask , (margin-tmp_dist)*(margin-tmp_dist) , mask_zeros )

        # error = torch.sum(  masked_dist_zero ) /(n_core*(n_core-1))
        delta_dist = (margin-tmp_dist)**2
        error = torch.mean( delta_dist )

        max_error = torch.zeros(1)    
        for i in range( n_core ):
            max_id = torch.argmax( delta_dist[i] , dim = 0 )
            max_error += delta_dist[i][max_id]
        max_error /= n_core

        loss = error + max_error

        loss.backward()

        optimizer.step()
        scheduler.step()
        if ( epoch % epoch_iter == epoch_iter -1 ):
            print('loss =  %.5f  error = %.5f ' % ( loss.item() , error.item() ) )
    core_norm = torch.norm( core , dim=1 , p =2 )
    print('before Normalization  max = %.3f min=%.3f'%(core_norm.max() , core_norm.min() ))
    print('Finished.')
    core.data /= core_norm.view(-1,1)
    core_norm = torch.norm( core , dim=1 , p =2 )
    print('after Normalization  max = %.3f min=%.3f'%(core_norm.max() , core_norm.min() ))

    core_sim = get_cos_dist( core , core )
    print( torch.mean( core_sim , dim=1  ) )
    # out_float_tensor(core_sim)
    # print('sum cos dist')
    # sum = torch.sum(core_sim , dim = 1)
    # out_float_tensor(sum)
    return core


def get_core(argv):
    n_label = 345
    dim = 512

    core = generate_core(n_label,dim)
    core = core.detach().numpy()
    np.save('core.npy',core)

if __name__ == '__main__':
    get_core(argv)