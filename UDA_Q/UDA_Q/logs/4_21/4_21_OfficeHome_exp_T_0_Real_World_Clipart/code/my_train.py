from audioop import avg
from cv2 import split
from numpy import zeros_like
from sqlalchemy import true
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from icecream import ic
import time
from tqdm import tqdm
from tool import Avg_er
import sys

sys.path.append('..')
from UDA_Q.loss import CB_predict_loss,Balance_loss,Q_gather_loss,trust_loss
from fixed_dataset import fixed_dataset
from my_dataset import ImageDataset

def train_source( S_train_loader, model,list_optimizer,quantizer=None,logger=None ):
    model.Is_source = True

    mean_loss = Avg_er('mean_loss')
    mean_loss_CB_cls = Avg_er('mean_loss_CB_cls')
    mean_loss_cls = Avg_er('mean_loss_cls')
    mean_loss_Balance = Avg_er('mean_loss_Balance')

    mean_mse = Avg_er('mse')

    from pseudo_label import generate_feature
    
    with torch.no_grad():
        model.eval()  #在4.8 及之前的代码没有这一行
        S_f,S_label = generate_feature(S_train_loader,model)
        # if type(train_loader1.dataset) == fixed_dataset:
        #     base_f = train_loader1.dataset.f.cuda()
        #     S_f = generate_feature(base_f,model)
        #     S_label = torch.LongTensor(train_loader1.dataset.label).cuda()
        # elif type(train_loader1.dataset) == ImageDataset:
        #     S_f = generate_feature(,model)
        prob_CB_cls = quantizer.get_prob_CB_cls(S_f,S_label)

    model.train()

    CE_loss = torch.nn.CrossEntropyLoss()

    # for n_iter, (S_base_f,S_label,S_idx) in enumerate(train_loader1):
    for n_iter, data in enumerate(S_train_loader):
        S_base_f,S_label = data
        for optimizer in list_optimizer:
            optimizer.zero_grad()

        S_base_f = S_base_f.cuda()
        S_label = S_label.cuda()
        # S_feat,S_predict = model(S_base_f,'Source')
        S_feat,S_predict = model(S_base_f)

        loss_cls = CE_loss(S_predict, S_label)
        
        
        loss_CB_cls = CB_predict_loss(S_feat,S_label,quantizer,prob_CB_cls)

        loss_Balance = Balance_loss(S_feat,quantizer)
        
        Q_S_feat,_ = quantizer( S_feat)
        mse = torch.norm( Q_S_feat-S_feat, p=2,dim=1).mean()

        # loss = loss_cls + loss_CB_cls + 10*loss_Balance
        loss = loss_CB_cls + 10*loss_Balance
        
        loss.backward()

        for optimizer in list_optimizer:
            optimizer.step()
        
        mean_loss.add( loss.item() , S_base_f.size(0) )
        mean_loss_cls.add( loss_cls.item() , S_base_f.size(0) )
        mean_loss_CB_cls.add( loss_CB_cls.item() , S_base_f.size(0) )
        mean_loss_Balance.add( loss_Balance.item() , S_base_f.size(0) )
        
        mean_mse.add( mse.item() , S_base_f.size(0) )

    loss_output =  ' '.join( [  mean_loss.out_s(),mean_loss_cls.out_s(), 
                                mean_loss_CB_cls.out_s(), mean_loss_Balance.out_s() ])
    logger.info(loss_output)
    aux_output =  ' '.join( [mean_mse.out_s()])
    logger.info(aux_output)
    return mean_loss.mean

def train_S_T_pair( epoch , core ,train_loader,T_train_set, model, list_optimizer, quantizer=None,logger=None ):
    mean_loss = Avg_er()
    mean_S_loss_gather = Avg_er()
    mean_T_loss_gather = Avg_er()
    
    mean_S_loss_cls = Avg_er()
    mean_T_loss_cls = Avg_er()

    mean_loss_mse = Avg_er()

    mean_loss_joint = Avg_er()

    acc_pseudo_label = Avg_er()

    model.train()

    CE_loss = torch.nn.CrossEntropyLoss()
    for n_iter, (S_base_f,T_base_f,S_label,T_pseudo_label,S_idx,T_idx) in enumerate(train_loader):
        for optimizer in list_optimizer:
            optimizer.zero_grad()

        S_base_f = S_base_f.cuda()
        T_base_f = T_base_f.cuda()
        S_label = S_label.cuda()
        T_pseudo_label = T_pseudo_label.cuda()

        S_feat,S_predict = model(S_base_f)
        T_feat,T_predict = model(T_base_f)


        S_loss_gather = torch.norm( S_feat-core[S_label] , p=2,dim=1).mean()
        # S_loss_gather = torch.zeros(1).cuda()

        T_loss_gather = torch.norm( T_feat-core[T_pseudo_label] , p=2,dim=1).mean()
        # T_loss_gather = torch.zeros(1).cuda()

        
        S_loss_CE = CE_loss(S_predict, S_label)
        # S_loss_CE = torch.zeros_like(S_loss_gather)
        
        T_loss_CE = CE_loss(T_predict, T_pseudo_label)
        # T_loss_CE = torch.zeros_like(T_loss_gather)

        if quantizer == None:
            loss_joint = torch.norm(T_feat-S_feat,p=2,dim=1).mean()
        else:
            Q_S_feat,_= quantizer(S_feat,5)
            loss_joint_1 = torch.norm(T_feat-S_feat,p=2,dim=1).mean()
            loss_joint_2 = torch.norm(T_feat-Q_S_feat,p=2,dim=1).mean()
            loss_joint = loss_joint_1 + 0.1*loss_joint_2

            mean_loss_mse.add( torch.norm(S_feat-Q_S_feat,p=2,dim=1).mean() , S_base_f.size(0) )
        
        loss = (S_loss_gather + T_loss_gather)  + ( S_loss_CE + T_loss_CE) + loss_joint

        loss.backward()

        for optimizer in list_optimizer:
            optimizer.step()


        mean_loss.add( loss.item() , S_base_f.size(0) )
        mean_S_loss_gather.add( S_loss_gather.item() , S_base_f.size(0) )
        mean_T_loss_gather.add( T_loss_gather.item() , T_base_f.size(0) )

        mean_S_loss_cls.add( S_loss_CE.item() , S_base_f.size(0) )
        mean_T_loss_cls.add( T_loss_CE.item() , S_base_f.size(0) )

        mean_loss_joint.add( loss_joint.item() , S_base_f.size(0) )

        acc = torch.mean( ( S_label == torch.LongTensor(T_train_set.label[ T_idx ]).cuda() ).float() )
        acc_pseudo_label.add( acc.item() , S_base_f.size(0))
    # print(f'loss = {mean_loss.mean:.3f} \
    logger.info(f'loss = {mean_loss.mean:.3f} \
        S_loss_g = {mean_S_loss_gather.mean:.3f} S_loss_cls = {mean_S_loss_cls.mean:.3f} \
        T_loss_g = {mean_T_loss_gather.mean:.3f} T_loss_cls = {mean_T_loss_cls.mean:.3f} \
        mean_mse = {mean_loss_mse.mean:.3f} ')
    # print(f'acc_pseudo_label = {acc_pseudo_label.mean:.3f} mean_loss_joint = {mean_loss_joint.mean:.3f}')
    logger.info(f'acc_pseudo_label = {acc_pseudo_label.mean:.3f} mean_loss_joint = {mean_loss_joint.mean:.3f}')

def train_T_sample(T_train_loader,S_train_loader, backbone, list_optimizer , quantizer,logger ):
    backbone.Is_source = False

    mean_loss = Avg_er('mean_loss')
    mean_loss_retrieve = Avg_er('mean_loss_retrieve')
    mean_loss_cls = Avg_er('mean_loss_cls')

    mean_mse = Avg_er('mse')

    from pseudo_label import generate_feature
    
    with torch.no_grad():
        backbone.eval()
        S_f,S_label = generate_feature(S_train_loader,backbone)
        prob_CB_cls = quantizer.get_prob_CB_cls(S_f,S_label)

    backbone.train()

    CE_loss = torch.nn.CrossEntropyLoss()

    grad_sum = 0
    
    for n_iter, (T_base_f,T_pseudo_label) in enumerate(T_train_loader):
        for optimizer in list_optimizer:
            optimizer.zero_grad()
        batch_size = T_base_f.size(0)

        T_base_f = T_base_f.cuda()
        T_pseudo_label = T_pseudo_label.cuda()
        # T_feat,T_predict = backbone(T_base_f,source_flag=False)
        # T_feat,T_predict = backbone(T_base_f,'Target')
        T_feat,T_predict = backbone(T_base_f)

        loss_cls = CE_loss(T_predict, T_pseudo_label)
        
        # rand_S_id = torch.randint( 0, int(1e9+6), ( batch_size,) ).cuda()
        # rand_S_id = rand_S_id % cls_cnt[T_pseudo_label]
        # S_code = code_memory[ T_pseudo_label , rand_S_id ]
        
        # loss_retrieve = Q_gather_loss(T_feat,S_code,quantizer)
        # loss_retrieve = torch.zeros_like(loss_cls)
        loss_retrieve = trust_loss(T_feat,quantizer,prob_CB_cls)

        # loss = loss_cls + 1*loss_retrieve
        # loss = loss_retrieve
        loss = loss_retrieve
        
        loss.backward()

        for optimizer in list_optimizer:
            optimizer.step()
        
        mean_loss.add( loss.item() , batch_size )
        mean_loss_cls.add( loss_cls.item() , batch_size )
        mean_loss_retrieve.add( loss_retrieve.item() , batch_size )

    loss_output =  ' '.join( [  mean_loss.out_s(),mean_loss_cls.out_s(), 
                                mean_loss_retrieve.out_s() ])
    logger.info(loss_output)
    return mean_loss.mean
    