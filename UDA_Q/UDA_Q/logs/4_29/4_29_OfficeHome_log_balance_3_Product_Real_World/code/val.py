from audioop import avg
from tabnanny import check
from cv2 import sort, split
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

from fixed_dataset import fixed_dataset
from fixed_dataset import split_dataset,split_DomainNet,split_OfficeHome

from evaluate import test_mAP

from processor.processor_uda import compute_knn_idx,generate_new_dataset,train_collate_fn,update_feat
from tool import Avg_er

import logging

from quantizers.Kmeans_PQ import Kmeans_PQ
from quantizers.prob_quantizer import Prob_Quantizer

from base_models import model_fc
from my_train import train_source,train_S_T_pair
from tqdm import tqdm
from pseudo_label import JS_dist,p_2_pseudo_label,get_prob_codebook
from pseudo_label import generate_feature

# def generate_feature(base_f,label,backbone):
#     backbone.eval()

#     f = torch.FloatTensor( base_f.size(0) , 512 ).cuda()
#     predict_label = torch.LongTensor( base_f.size(0) ).cuda()
#     predict_prob = torch.FloatTensor( base_f.size(0) ).cuda()

#     bs = 64
#     cnt = 0
#     acc = 0
#     with torch.no_grad():
#         for st in tqdm( range( 0 , base_f.size(0) , bs ) ):
#             ed = min(st + bs , base_f.size(0))
#             batch_base_f = base_f[st:ed]
#             f[ st:ed] , predict = backbone(batch_base_f)

#             predict_prob[st:ed], predict_label[st:ed] = torch.max( predict, dim = 1)

#             cnt += (ed-st)
#     assert( cnt == base_f.size(0) )
    
#     acc = (predict_label == label).float().mean()
#     print('predictor acc = ',acc)
#     return f

def draw_3D(data,name):
    import matplotlib.pyplot as plt
    import numpy as np
    data= data.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111, projection='3d')
    _x = np.arange(data.shape[0])
    _y = np.arange(data.shape[1])
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = data.ravel()
    bottom = np.zeros_like(top)
    width = depth = 0.35

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    plt.savefig(name)

def write_tensor_npy(x,dir):
    fp = open(dir,'w')
    for i in range(x.size(0)):
        fp.write( f'{x[i].item():.3f} ')
    fp.close()


def test_Q_pseudo_label(T_f,T_label,quantizer,logger):

    n_f = T_f.size(0)

    p_code_cls = get_prob_codebook(T_f,T_label.cuda(),quantizer)
    prob_cls = p_code_cls[0].sum(dim=1)
    prob_code = p_code_cls[0].sum(dim=0)
    # print(prob_cls.size())
    # print(prob_cls)

    # print(prob_code.size())
    # print(prob_code)
    # exit()
    # tmp = p_code_cls[0]
    # draw_3D(tmp.data,'P_cod_cls_balanced.pdf')
    # exit()
    MAX_p,_ = torch.max( p_code_cls, dim =2)
    logger.info(f'MAX_p_code_cls_mean = {MAX_p.mean():.3f} ')

    pseudo_label,prob_f_cls = p_2_pseudo_label( T_f , quantizer, p_code_cls,logger )
    
    acc = (pseudo_label == T_label).float().mean()
    logger.info(f'p_2_pseudo_label acc = {acc:.3f}')


    prob_f_cls,_ = torch.max(prob_f_cls,dim=1)

    sort_idx = torch.argsort( prob_f_cls , descending=True )

    mid_idx = sort_idx[ sort_idx.size(0)//2 ]
    part_idx = sort_idx[ int(sort_idx.size(0)*0.3) ]
    logger.info(f' max prob = {prob_f_cls[ sort_idx[0] ]:.3f}')
    logger.info(f' mid prob = {prob_f_cls[ mid_idx ]:.3f}')
    logger.info(f' part prob = {prob_f_cls[ part_idx ]:.3f}')

    # write_tensor_txt( prob_f_cls[ sort_idx ], '../vis/prob.txt' )
    # OfficeHome DomainNet
    # np.save( '../vis/prob_'+'DomainNet'+'.npy',prob_f_cls[ sort_idx ].detach().cpu().numpy())
    # exit()
    
    
    # for div in [0.1, 0.3,1.0]:
    for div in [0.1, 0.3 , 0.5 , 0.7 ,1]:
        n_part = int( n_f*div )
        val_idx = sort_idx[ : n_part ]

        fine_pseudo_label = pseudo_label[ val_idx ]
        fine_T_label = T_label[ val_idx ]
        acc = (fine_pseudo_label == fine_T_label).float().mean()
        logger.info(f'fine_idx @{div:.3f} acc = {acc:.3f}')
        
    logger.info('\n\n')

def get_logger():
    logger = logging.getLogger('UDA_Q_log')
    logger.setLevel(level = logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',"%Y-%m-%d-%H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger

def test_mse(f,Q):
    bs = 128
    cnt = 0
    mse = 0
    with torch.no_grad():
        for st in tqdm( range( 0 , f.size(0) , bs ) ):
            ed = min(st + bs , f.size(0))
            batch_f = f[st:ed]
            Q_f,_ = Q(batch_f)
            mse += torch.norm( Q_f-batch_f , dim =1 ).sum().item()
            cnt += (ed-st)
    assert( cnt == f.size(0) )
    mse /= cnt
    return mse

# def test_PL(f,Q):
#     bs = 128
#     cnt = 0
#     mse = 0
#     PL = torch.FloatTensor( Q.n_quantizer, Q.n_codeword ).cuda()
#     with torch.no_grad():
#         for st in tqdm( range( 0 , f.size(0) , bs ) ):
#             ed = min(st + bs , f.size(0))
#             batch_f = f[st:ed]
#             cnt += (ed-st)
#     assert( cnt == f.size(0) )
#     mse /= cnt

if __name__ == '__main__':
    logger = get_logger()

    dataset_dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/domainnet/'
    dataset = 'OfficeHome'

    if dataset == 'OfficeHome':    
        seen_domain = 'Product'
        unseen_domain = 'Real_World'
        n_class = 65
        n_quantizer = 8
        _, S_query_set, S_train_set = split_OfficeHome(seen_domain)
        S_gallery_set = S_train_set

        T_train_set, T_query_set, _ = split_OfficeHome(unseen_domain)
        T_gallery_set = T_train_set
    elif dataset == 'DomainNet':  
        n_quantizer = 4
        n_class = 345
        seen_domain = 'real'
        unseen_domain = 'clipart'
        S_train_set, S_query_set, S_gallery_set = split_DomainNet(seen_domain)
        T_train_set, T_query_set, T_gallery_set = split_DomainNet(unseen_domain)

    ic(S_train_set.__len__())
    ic(S_query_set.__len__())
    ic(S_gallery_set.__len__())

    ic(T_train_set.__len__())
    ic(T_query_set.__len__())
    ic(T_gallery_set.__len__())

    test_name = 'OfficeHome'

    if test_name == 'OfficeHome':
        # model_path = '/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/logs/4_18/4_18_OfficeHome_1_Real_World_Product/model/4_18_OfficeHome_1_80.pth'
        model_path = '/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/logs/4_20/4_20_OfficeHome_0_Product_Real_World/model/4_20_OfficeHome_0_60.pth'
    elif test_name == 'DomainNet':
        model_path = '/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/logs/4_7/4_7_mine_CE_Balance_MSE_6_real_clipart/model/4_7_mine_CE_Balance_MSE_6_55.pth'
    else:
        assert(False)

    dim = 512
    quantizer = Prob_Quantizer(n_quantizer,256,dim).cuda()
    top_k = -1
    backbone = model_fc(dim,n_class).cuda()
    checkpoint = torch.load(model_path)
    backbone.load_state_dict( checkpoint['backbone_state_dict'] )
    backbone.eval()

    quantizer.load_state_dict( checkpoint['quantizer_state_dict'])

    batch_size = 64
    num_workers = 4
    S_train_loader = DataLoader( S_train_set, batch_size=batch_size, shuffle=True,num_workers=num_workers, pin_memory=True)

    T_train_loader = DataLoader( T_train_set, batch_size=batch_size, shuffle=True,num_workers=num_workers, pin_memory=True)

    
    S_f,S_label = generate_feature(S_train_loader,backbone)
    T_f,T_label = generate_feature(T_train_loader,backbone)

    mse = test_mse(S_f,quantizer)
    logger.info(f'mse =  {mse:.3f}')

    test_Q_pseudo_label(T_f,T_label,quantizer,logger)
    test_Q_pseudo_label(S_f,S_label,quantizer,logger)
    exit()
    logger.info(f'single domain = ')
    test_mAP(S_query_set,S_gallery_set,True,True,backbone,Q=quantizer,top_k=100,dim=512,logger=logger)
    
    logger.info(f'cross domain = ')
    test_mAP(T_query_set,S_gallery_set,False,True,backbone,Q=quantizer,top_k=100,dim=512,logger=logger)

    # print(quantizer.CodeBooks[0][0])
    exit()

    from quantizers.Kmeans_PQ import Kmeans_PQ
    
    K_PQ = Kmeans_PQ(n_quantizer,256,dim)
    K_PQ.train(all_im_em=S_f.detach().cpu().numpy())

    mse = test_mse(S_f,K_PQ)
    logger.info(f'kmeans mse =  {mse:.3f}')
    # exit()
    with torch.no_grad():
        for i in range(4):
            quantizer.CodeBooks[i] = K_PQ.list_Q[i].CodeBook

    test_Q_pseudo_label(T_f,T_label,quantizer,logger)
    logger.info(f'single domain = ')
    test_mAP(S_query_set,S_gallery_set,True,True,backbone,Q=quantizer,top_k=100,dim=512,logger=logger)
    
    logger.info(f'cross domain = ')
    test_mAP(T_query_set,S_gallery_set,False,True,backbone,Q=quantizer,top_k=100,dim=512,logger=logger)