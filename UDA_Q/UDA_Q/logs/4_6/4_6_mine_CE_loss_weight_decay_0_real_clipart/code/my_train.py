from audioop import avg
from cv2 import split
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
from UDA_Q.loss import CB_predict_loss


def train_source( epoch , core ,train_loader1, model,list_optimizer,quantizer=None,logger=None ):
    mean_loss = Avg_er('mean_loss')
    mean_loss_CB_cls = Avg_er('mean_loss_CB_cls')
    mean_loss_cls = Avg_er('mean_loss_cls')

    mean_mse = Avg_er('mse')

    from pseudo_label import generate_feature
    
    with torch.no_grad():
        base_f = train_loader1.dataset.f.cuda()
        S_f = generate_feature(base_f,model)
        S_label = torch.LongTensor(train_loader1.dataset.label).cuda()
        prob_CB_cls = quantizer.get_prob_CB_cls(S_f,S_label)

    model.train()

    loss_CE = torch.nn.CrossEntropyLoss()

    for n_iter, (S_img,S_label,S_idx) in enumerate(train_loader1):
        for optimizer in list_optimizer:
            optimizer.zero_grad()

        S_img = S_img.cuda()
        S_label = S_label.cuda()
        S_feat,S_predict = model(S_img)

        loss_cls = loss_CE(S_predict, S_label)
        
        
        loss_CB_cls = CB_predict_loss(S_feat,S_label,quantizer,prob_CB_cls)
        
        Q_S_feat,_ = quantizer( S_feat)
        mse = torch.norm( Q_S_feat-S_feat, p=2,dim=1).mean()

        loss = loss_cls + loss_CB_cls
        
        loss.backward()

        for optimizer in list_optimizer:
            optimizer.step()
        
        # quantizer.Intra_Normalize()

        mean_loss.add( loss.item() , S_img.size(0) )
        mean_loss_cls.add( loss_cls.item() , S_img.size(0) )
        mean_loss_CB_cls.add( loss_CB_cls.item() , S_img.size(0) )
        mean_mse.add( mse.item() , S_img.size(0) )

    loss_output =  ' '.join( [mean_loss.out_s(),mean_loss_cls.out_s(),mean_loss_CB_cls.out_s()])
    logger.info(loss_output)
    aux_output =  ' '.join( [mean_mse.out_s()])
    logger.info(aux_output)

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

    loss_CE = torch.nn.CrossEntropyLoss()
    for n_iter, (S_img,T_img,S_label,T_pseudo_label,S_idx,T_idx) in enumerate(train_loader):
        for optimizer in list_optimizer:
            optimizer.zero_grad()

        S_img = S_img.cuda()
        T_img = T_img.cuda()
        S_label = S_label.cuda()
        T_pseudo_label = T_pseudo_label.cuda()

        S_feat,S_predict = model(S_img)
        T_feat,T_predict = model(T_img)


        S_loss_gather = torch.norm( S_feat-core[S_label] , p=2,dim=1).mean()
        # S_loss_gather = torch.zeros(1).cuda()

        T_loss_gather = torch.norm( T_feat-core[T_pseudo_label] , p=2,dim=1).mean()
        # T_loss_gather = torch.zeros(1).cuda()

        
        S_loss_CE = loss_CE(S_predict, S_label)
        # S_loss_CE = torch.zeros_like(S_loss_gather)
        
        T_loss_CE = loss_CE(T_predict, T_pseudo_label)
        # T_loss_CE = torch.zeros_like(T_loss_gather)

        if quantizer == None:
            loss_joint = torch.norm(T_feat-S_feat,p=2,dim=1).mean()
        else:
            Q_S_feat,_= quantizer(S_feat,5)
            loss_joint_1 = torch.norm(T_feat-S_feat,p=2,dim=1).mean()
            loss_joint_2 = torch.norm(T_feat-Q_S_feat,p=2,dim=1).mean()
            loss_joint = loss_joint_1 + 0.1*loss_joint_2

            mean_loss_mse.add( torch.norm(S_feat-Q_S_feat,p=2,dim=1).mean() , S_img.size(0) )
        
        loss = (S_loss_gather + T_loss_gather)  + ( S_loss_CE + T_loss_CE) + loss_joint

        loss.backward()

        for optimizer in list_optimizer:
            optimizer.step()


        mean_loss.add( loss.item() , S_img.size(0) )
        mean_S_loss_gather.add( S_loss_gather.item() , S_img.size(0) )
        mean_T_loss_gather.add( T_loss_gather.item() , T_img.size(0) )

        mean_S_loss_cls.add( S_loss_CE.item() , S_img.size(0) )
        mean_T_loss_cls.add( T_loss_CE.item() , S_img.size(0) )

        mean_loss_joint.add( loss_joint.item() , S_img.size(0) )

        acc = torch.mean( ( S_label == torch.LongTensor(T_train_set.label[ T_idx ]).cuda() ).float() )
        acc_pseudo_label.add( acc.item() , S_img.size(0))
    # print(f'loss = {mean_loss.mean:.3f} \
    logger.info(f'loss = {mean_loss.mean:.3f} \
        S_loss_g = {mean_S_loss_gather.mean:.3f} S_loss_cls = {mean_S_loss_cls.mean:.3f} \
        T_loss_g = {mean_T_loss_gather.mean:.3f} T_loss_cls = {mean_T_loss_cls.mean:.3f} \
        mean_mse = {mean_loss_mse.mean:.3f} ')
    # print(f'acc_pseudo_label = {acc_pseudo_label.mean:.3f} mean_loss_joint = {mean_loss_joint.mean:.3f}')
    logger.info(f'acc_pseudo_label = {acc_pseudo_label.mean:.3f} mean_loss_joint = {mean_loss_joint.mean:.3f}')

def train_cls(S_train_set,T_train_set,S_query_set,T_query_set,model):
    train_epoch = 100
    batch_size = 64
    from my_train import Avg_er

    CEloss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)

    train_loader1 = DataLoader( S_train_set, batch_size=batch_size,shuffle=False,num_workers=4, pin_memory=True)

    S_query_loader = DataLoader( S_query_set, batch_size=batch_size,shuffle=False,num_workers=4, pin_memory=True)
    T_query_loader = DataLoader( T_query_set, batch_size=batch_size,shuffle=False,num_workers=4, pin_memory=True)


    for epoch in range(train_epoch):
        avg_loss = Avg_er()
        model.train()
        for iter, data in enumerate(train_loader1,0):
            optimizer.zero_grad()
        
            org_vec,labels = data[0],data[1]
            org_vec = org_vec.cuda()
            labels = labels.cuda()

            org_vec = org_vec + torch.normal( mean = torch.zeros_like(org_vec) , std = 1e-3*torch.ones_like(org_vec) )

            _,(f,_,predict) = model(org_vec,return_feat_prob=True)

            loss = CEloss(predict , labels)
            loss.backward()

            avg_loss.add( loss.item() , org_vec.size(0) )

            optimizer.step()
        print(f'loss = {avg_loss.mean:.3f}')

        if epoch%10 == 0:
            model.eval()
            mean_acc = Avg_er()
            for iter, data in enumerate(S_query_loader,0):
                org_vec,labels = data[0],data[1]
                org_vec = org_vec.cuda()
                labels = labels.cuda()

                _,(f,_,predict) = model(org_vec,return_feat_prob=True)
                predict_label = torch.argmax(predict,dim=1)
                
                mean_acc.add( (predict_label==labels).long().mean() , org_vec.size(0) )

            print(f'======          prec = {mean_acc.mean:.3f}')