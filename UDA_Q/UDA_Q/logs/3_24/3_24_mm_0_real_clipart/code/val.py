from audioop import avg
from tabnanny import check
from cv2 import split
from sqlalchemy import true
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import sys
from icecream import ic
import os
from domainnet_options import Options

sys.path.append('..')
# from config import cfg
from utils.logger import setup_logger

from tool import Avg_er

from fixed_dataset import fixed_dataset,pair_dataset
from fixed_dataset import split_dataset,pair_train_collate_fn

from evaluate import test_mAP

from processor.processor_uda import compute_knn_idx,generate_new_dataset,train_collate_fn,update_feat
from tool import Avg_er

import logging

from quantizers.Kmeans_PQ import Kmeans_PQ

from base_models import model_fc
from my_train import Quantizer
from my_train import train_source,train_S_T_pair
from tqdm import tqdm

def generate_feature(base_f,backbone):
    f = torch.FloatTensor( base_f.size(0) , dim ).cuda()

    bs = 64
    cnt = 0
    for st in tqdm( range( 0 , base_f.size(0) , bs ) ):
        ed = min(st + bs , base_f.size(0))
        batch_base_f = base_f[st:ed]
        _,f[ st:ed] = backbone(batch_base_f)
        cnt += (ed-st)
    assert( cnt == base_f.size(0) )
    return f

if __name__ == '__main__':
    dataset_dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/domainnet/'
    seen_domain = 'real'
    unseen_domain = 'clipart'
    S_train_set, S_query_set, S_gallery_set = split_dataset( dataset_dir+seen_domain+'/', False)
    T_train_set, T_query_set, T_gallery_set = split_dataset( dataset_dir+unseen_domain+'/', False)

    model_path = '/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/logs/3_21_Q_0_real_clipart/model/3_21_Q_0_20.pth'

    quantizer = Quantizer(4,256,512).cuda()
    dim = 512
    n_class = 345
    backbone = model_fc(dim,n_class).cuda()
    checkpoint = torch.load(model_path)
    backbone.load_state_dict( checkpoint['backbone_state_dict'] )
    backbone.eval()

    quantizer.load_state_dict( checkpoint['quantizer_state_dict'])

    S_base_f = S_train_set.f.cuda()
    S_label = torch.LongTensor( S_train_set.label ).cuda()

    T_base_f = T_train_set.f.cuda()
    T_label = torch.LongTensor( T_train_set.label ).cuda()

    
    S_f = generate_feature(S_base_f,backbone)
    T_f = generate_feature(T_base_f,backbone)

    
    bs = 64

    dist_pair = torch.zeros( S_f.size(0) ).cuda()
    true_pair = torch.zeros( S_f.size(0) ).cuda()

    sub_codebook = quantizer.CodeBooks[0]
    
    p_T_f = torch.softmax( torch.cdist( T_f[:,:128] , sub_codebook ) , dim = 1)

    check_id = torch.zeros( T_f.size(0) )

    with torch.no_grad():
        for st in tqdm(range( 0 , S_f.size(0) , bs )):
            ed = min(st + bs , S_f.size(0))
            batch_S_f = S_f[st:ed]
            batch_S_label = S_label[st:ed]

            batch_p_S_f = torch.softmax( torch.cdist( batch_S_f[:,:128] , sub_codebook) , dim = 1)

            kl_loss = torch.nn.KLDivLoss(reduction = "none")
            kl_dist = torch.zeros( batch_p_S_f.size(0) , T_f.size(0) )
            for i in range(ed-st):
                mid = (batch_p_S_f[i].view(1,-1) + p_T_f)/2
                
                tmp = kl_loss( batch_p_S_f[i].view(1,-1).log() , mid ).sum(dim=1) + kl_loss( p_T_f, mid ).sum(dim=1)

                # tmp = kl_loss( batch_p_S_f[i].view(1,-1) , p_T_f ).sum(dim=1)
                # tmp = ( batch_p_S_f[i].view(1,-1) * p_T_f ).sum(dim=1)
                # tmp = torch.cdist(batch_p_S_f[i].view(1,-1) , p_T_f )
                # print(tmp.size())
                kl_dist[i] = tmp
                # exit()

            dist = torch.cdist( batch_S_f , T_f )

            dist_pair[st:ed],MIN_id = torch.min( dist , dim = 1)
            # dist_pair[st:ed],MIN_id = torch.min( kl_dist , dim = 1)
            
            true_label = T_label[MIN_id]
            
            check_id[MIN_id] =1

            true_pair[st:ed] = (true_label==batch_S_label).float()
    print(true_pair.size())
    print(check_id.size())
    print(check_id.sum())
    new_id = torch.argsort( dist_pair ,dim=0)
    dist_pair = dist_pair[new_id]
    true_pair = true_pair[new_id]
    
    acc = torch.cumsum( true_pair , dim=0 ) / (torch.FloatTensor( range( true_pair.size(0) )).cuda()+1)
    np.save('./acc_dist_Q_60.npy', acc.cpu().detach().numpy() )
    print(f'pseudo_label acc = {true_pair.mean():.3f}')

    half_true_pair = true_pair[ : S_f.size(0)//2 ]
    print(f'pseudo_label acc of first half = {half_true_pair.mean():.3f}')

    half_true_pair = true_pair[ S_f.size(0)//2 : ]
    print(f'pseudo_label acc of second half = {half_true_pair.mean():.3f}')