from audioop import avg
from cv2 import split
from sqlalchemy import true
from sympy import Q
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
# from utils.logger import setup_logger


from quantizers.prob_quantizer import Prob_Quantizer

import logging


from fixed_dataset import split_OfficeHome,split_DomainNet,get_MNIST_USPS

from my_dataset import get_CIFAR10, get_ImageNet, get_img_DomainNet,get_img_MNIST_USPS

from evaluate import test_mAP

from processor.processor_uda import compute_knn_idx,generate_new_dataset,train_collate_fn,update_feat
from tool import Avg_er

import logging

from quantizers.Kmeans_PQ import Kmeans_PQ

from base_models import model_fc, model_vgg,model_AlexNet_single_head,model_S_T_fc,model_vgg_single_head
from base_models import model_simple_fc,model_fc_cls,model_refix_fc,model_vgg_bn_all
from base_models import model_fc_MNIST
from my_train import train_source,train_T_sample

from pseudo_label import NN_pseudo_label,prob_pseudo_label


def collect_data(backbone,data_loader):
    all_f = None
    for (base_f,label) in data_loader:
        label = label.cuda()
        base_f = base_f.cuda()

        label = label.view(-1)

        out_f,_ = backbone(base_f)
        if all_f is None:
            all_f = out_f
            all_label = label
        else:
            all_f = torch.cat( (all_f,out_f),dim=0)
            all_label = torch.cat( (all_label,label),dim=0)
    print(all_f.size())
    return all_f,all_label
        

def set_logger():
    logger = logging.getLogger('UDA_Q_log')
    logger.setLevel(level = logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s',"%Y-%m-%d-%H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger

class Tmp_args:
    def __init__(self,):
        self.seen_domain = 'Real_World'
        self.unseen_domain = 'Product'
        self.n_class = 65

if __name__ == '__main__':
    logger = set_logger()
    args = Tmp_args()
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    dim = 128
    n_class = 65
    num_quantzer = 8
    n_codeword = 256
    print('dim = '+str(dim))

    _, S_query_set, S_train_set = split_OfficeHome(args.seen_domain)
    S_gallery_set = S_train_set

    T_train_set, T_query_set, _ = split_OfficeHome(args.unseen_domain)

    
    top_k = -1
    backbone = model_fc(dim,n_class).cuda()

    ic(S_train_set.__len__())
    ic(S_query_set.__len__())
    ic(S_gallery_set.__len__())


    ic(T_train_set.__len__())
    ic(T_query_set.__len__())

    quantizer = Prob_Quantizer(num_quantzer,n_codeword,dim)
    quantizer = quantizer.cuda()

    load_model_path = '/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/logs/5_05/5_05_OfficeHome_fin_4_Real_World_Product/model/5_05_OfficeHome_fin_4_60.pth'
    print('load from '+load_model_path)
    checkpoint = torch.load(load_model_path)
    backbone.load_state_dict( checkpoint['backbone_state_dict'] )
    backbone.eval()

    quantizer.load_state_dict( checkpoint['quantizer_state_dict'])
    print(f'test single-domain = ')
    test_mAP(args,S_query_set, S_gallery_set,True,True,backbone,top_k=top_k, dim=dim, logger=logger)
        
    print(f'test cross-domain = ')
    test_mAP(args,T_query_set, S_gallery_set,False,True, backbone,top_k=top_k, dim=dim, logger=logger)

    # flag_tsne = True
    flag_tsne = False
    
    flag_quan = True
    # flag_quan = False

    if flag_tsne:        
        S_train_loader = DataLoader( S_train_set, batch_size=64, shuffle=True,num_workers=4, pin_memory=True)
        T_train_loader = DataLoader( T_train_set, batch_size=64, shuffle=True,num_workers=4, pin_memory=True)
        
        S_f,S_label = collect_data(backbone,S_train_loader)
        T_f,T_label = collect_data(backbone,T_train_loader)

        S_label = S_label.view(-1)
        T_label = T_label.view(-1)

        torch.save(S_f,'./vis_data/S_f.npy')
        torch.save(S_label,'./vis_data/S_label.npy')
        torch.save(T_f,'./vis_data/T_f.npy')
        torch.save(T_label,'./vis_data/T_label.npy')

    if flag_quan:
        S_train_loader = DataLoader( S_train_set, batch_size=64, shuffle=True,num_workers=4, pin_memory=True)
        S_f,S_label = collect_data(backbone,S_train_loader)
        S_label = S_label.view(-1)

        prob_CB_cls = quantizer.get_prob_CB_cls(S_f,S_label,n_class)
        prob_CB_cls = prob_CB_cls[0]
        torch.save(prob_CB_cls,'./vis_data/prob_CB_cls.npy')

