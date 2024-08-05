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

from fixed_dataset import fixed_dataset
from fixed_dataset import split_dataset,split_DomainNet

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

def generate_feature(base_f,label,backbone):
    f = torch.FloatTensor( base_f.size(0) , 512 ).cuda()
    predict_label = torch.LongTensor( base_f.size(0) ).cuda()
    predict_prob = torch.FloatTensor( base_f.size(0) ).cuda()

    bs = 64
    cnt = 0
    acc = 0
    with torch.no_grad():
        for st in tqdm( range( 0 , base_f.size(0) , bs ) ):
            ed = min(st + bs , base_f.size(0))
            batch_base_f = base_f[st:ed]
            f[ st:ed] , predict = backbone(batch_base_f)

            predict_prob[st:ed], predict_label[st:ed] = torch.max( predict, dim = 1)

            cnt += (ed-st)
    assert( cnt == base_f.size(0) )
    
    acc = (predict_label == label).float().mean()
    print('predictor acc = ',acc)
    return f

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
    # draw_3D(tmp.data,'P_cod_cls.pdf')
    MAX_p,_ = torch.max( p_code_cls, dim =2)
    logger.info(f'MAX_p_code_cls_mean = {MAX_p.mean():.3f} ')

    pseudo_label,prob_f_cls = p_2_pseudo_label( T_f , quantizer, p_code_cls,logger )
    
    acc = (pseudo_label == T_label).float().mean()
    logger.info(f'p_2_pseudo_label acc = {acc:.3f}')


    prob_f_cls,_ = torch.max(prob_f_cls,dim=1)

    sort_idx = torch.argsort( prob_f_cls , descending=True )
    
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

    test_name = 'P_CB'

    if test_name == 'JS':
        model_path = '/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/logs/3_24/3_24_JS_0_real_clipart/model/3_24_JS_0_65.pth'
    elif test_name == 'L2':
        model_path = '/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/logs/3_31/3_31_0_real_clipart/model/3_31_0_85.pth'
    elif test_name == 'P_CB':
        model_path = '/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/logs/4_6/4_6_mine_CE_loss_real_clipart/model/4_6_mine_CE_loss_90.pth'
    else:
        assert(False)

    quantizer = Prob_Quantizer(4,256,512).cuda()
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

    
    S_f = generate_feature(S_base_f,S_label,backbone)
    T_f = generate_feature(T_base_f,T_label,backbone)

    mse = test_mse(S_f,quantizer)
    logger.info(f'mse =  {mse:.3f}')

    test_Q_pseudo_label(T_f,T_label,quantizer,logger)
    logger.info(f'single domain = ')
    test_mAP(S_query_set,S_gallery_set,backbone,Q=quantizer,top_k=100,dim=512,logger=logger)
    
    logger.info(f'cross domain = ')
    test_mAP(T_query_set,S_gallery_set,backbone,Q=quantizer,top_k=100,dim=512,logger=logger)

    print(quantizer.CodeBooks[0][0])

    from quantizers.Kmeans_PQ import Kmeans_PQ
    
    K_PQ = Kmeans_PQ(4,256,512)
    K_PQ.train(all_im_em=S_f.detach().cpu().numpy())

    mse = test_mse(S_f,K_PQ)
    logger.info(f'kmeans mse =  {mse:.3f}')
    # exit()
    with torch.no_grad():
        for i in range(4):
            quantizer.CodeBooks[i] = K_PQ.list_Q[i].CodeBook

    test_Q_pseudo_label(T_f,T_label,quantizer,logger)
    logger.info(f'single domain = ')
    test_mAP(S_query_set,S_gallery_set,backbone,Q=quantizer,top_k=100,dim=512,logger=logger)
    
    logger.info(f'cross domain = ')
    test_mAP(T_query_set,S_gallery_set,backbone,Q=quantizer,top_k=100,dim=512,logger=logger)
    exit()


    bs = 128

    dist_pair = torch.zeros( S_f.size(0) ).cuda()
    true_pair = torch.zeros( S_f.size(0) ).cuda()
    
    pseudo_label = torch.zeros( T_f.size(0) ).long().cuda() -1 
    cnt_T = torch.zeros( T_f.size(0)).cuda()
    

    cnt_multi_label = 0

    with torch.no_grad():
        for st in tqdm(range( 0 , S_f.size(0) , bs )):
            ed = min(st + bs , S_f.size(0))
            batch_S_f = S_f[st:ed]
            batch_S_label = S_label[st:ed]
            
            if test_name == 'JS':
                dist = JS_dist(batch_S_f , T_f , quantizer )
            elif test_name == 'L2':
                dist = torch.cdist( batch_S_f , T_f )
            else:
                assert(False)

            dist_pair[st:ed],MIN_id = torch.min( dist , dim = 1)
            
            true_label = T_label[MIN_id]
            
            true_pair[st:ed] = (true_label==batch_S_label).float()

            for i in range( MIN_id.size(0) ):
                cnt_T[ MIN_id[i] ] += 1
                if ( pseudo_label[ MIN_id[i] ] == -1 ):
                    pseudo_label[ MIN_id[i] ] = batch_S_label[i]
                elif ( pseudo_label[ MIN_id[i] ] != batch_S_label[i] ):
                    cnt_multi_label += 1
                    pseudo_label[ MIN_id[i] ] = -2
    
    ok_label = ( T_label ==  pseudo_label ).float().sum() / T_f.size(0)
    judged_T = ( pseudo_label > 0 ).float().sum()/ T_f.size(0)
    mean_acc = ok_label/judged_T

    ic(S_f.size())
    ic(T_f.size())
    ic( ok_label )
    ic( judged_T )
    ic( mean_acc )
    ic( cnt_multi_label)
    ic( cnt_T.sum() )
    ic( cnt_T.std() )
    ic( cnt_T.max() )

    cnt_T = cnt_T.long()
    # tmp_idx = torch.where(cnt_T > 0 )
    no_zero_cnt_T = cnt_T[ cnt_T >0 ]
    tmp_idx = torch.argsort(no_zero_cnt_T,dim=0,descending=True)
    sorted_cnt_T = no_zero_cnt_T[ tmp_idx ]

    bar_T = sorted_cnt_T
    ic(bar_T.sum())

    def mean_w_gap( data, gap ):
        n_data = torch.zeros( data.size(0)//gap+1 )
        for st in range(0,data.size(0),gap):
            ed = min(st+gap,data.size(0))
            n_data[st//gap] = data[st:ed].sum()/(ed-st)    
        return n_data

    gap = 50
    bar_T = mean_w_gap( bar_T , gap )
    # print(bar_T)
    # print(bar_T.size())
    np.save('./select_'+test_name+'.npy',bar_T.cpu().detach().numpy() )

    new_id = torch.argsort( dist_pair ,dim=0)
    dist_pair = dist_pair[new_id]
    true_pair = true_pair[new_id]
    
    # acc = torch.cumsum( true_pair , dim=0 ) / (torch.FloatTensor( range( true_pair.size(0) )).cuda()+1)
    gap = 2000
    acc = mean_w_gap( true_pair , gap )    
    # acc = torch.zeros( true_pair.size(0)//gap+1 )
    # for st in range(0,true_pair.size(0),gap):
    #     ed = min(st+gap,true_pair.size(0))
    #     acc[st//gap] = true_pair[st:ed].sum()/(ed-st)
    np.save('./vis_data/acc_dist_'+test_name+'.npy', acc.cpu().detach().numpy() )
    print(f'pseudo_label acc = {true_pair.mean():.3f}')

    half_true_pair = true_pair[ : S_f.size(0)//2 ]
    print(f'pseudo_label acc of first half = {half_true_pair.mean():.3f}')

    half_true_pair = true_pair[ S_f.size(0)//2 : ]
    print(f'pseudo_label acc of second half = {half_true_pair.mean():.3f}')