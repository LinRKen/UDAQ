
import time
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from sys import argv
import os
from icecream import ic

from models import Quantizer_DPgQ
from base_models import model_fc
from domainnet_options import Options
from train import do_train

import sys
sys.path.append('..')
from fixed_dataset import split_DomainNet
from evaluate import test_mAP

import logging

def set_logger(file_path):
    logger = logging.getLogger('DPgQ_log')
    logger.setLevel(level = logging.INFO)

    file_handler = logging.FileHandler(file_path,mode='w')
    formatter = logging.Formatter('%(asctime)s - %(message)s',"%Y-%m-%d-%H:%M:%S")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

if __name__ == '__main__':
    
    args = Options().parse()

    logger = set_logger(args.log_dir)

    logger.info('Parameters:\t' + str(args))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset == 'DomainNet':
        S_train_set, S_query_set, S_gallery_set = split_DomainNet(args.seen_domain)
        T_train_set, T_query_set, T_gallery_set = split_DomainNet(args.unseen_domain)

        args.n_class = 345
        dim = args.dim
        top_k = -1

        backbone = model_fc(n_class=args.n_class,dim=args.dim).cuda()
        quantizer = Quantizer_DPgQ(args.num_quantzer,args.n_codeword,args.dim).cuda()

        ic( S_train_set.__len__() )
        ic( T_query_set.__len__() )
        ic( S_gallery_set.__len__() )
    else:
        assert(False)

    
    train_loader = DataLoader(dataset=S_train_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    if args.optimizer == 'sgd':
        optimizer_f = optim.SGD( backbone.parameters(),lr=args.lr , momentum=args.momentum ,weight_decay=args.
        weight_decay)
    elif args.optimizer == 'adam':
        optimizer_f = optim.Adam( backbone.parameters(),lr=args.lr,betas=(0.5,0.5), weight_decay=args.weight_decay)
    # optimizer_Q = optim.Adam( quantizer.parameters()  , lr =args.Q_lr,betas=(0.5,0.5), weight_decay=args.Q_weight_decay)
    optimizer_Q = optim.Adam( quantizer.parameters()  , lr =args.Q_lr, weight_decay=args.Q_weight_decay)
    

    scheduler_f = lr_scheduler.StepLR(optimizer_f,step_size=10,gamma = 0.9  )
    scheduler_Q = lr_scheduler.ExponentialLR(optimizer_Q, 0.99)

    list_optimizer = [optimizer_f,optimizer_Q]
    list_scheduler = [scheduler_f,scheduler_Q]

    val_epoch = 10

    directory = args.model_dir
    save_name = '_'.join( (args.day , args.id) )

    best_map = -1
    wordvec = np.load('w2v_domainnet.npy', allow_pickle=True, encoding='latin1').item()
    class_name = np.load('name_list.npy').tolist()
    label_embeddings = torch.FloatTensor([wordvec.get(cl) for cl in class_name]).cuda()
    label_embeddings.requires_grad_(False)
    # test_mAP(query_set,gallery_set,backbone,Q=None,top_k=top_k,dim=300)
    # exit()
    warm_epoch = args.warm_epoch

    for epoch in range(1,args.epochs+1):
        if epoch<warm_epoch:
            do_train(epoch,train_loader,backbone,None,[optimizer_f],[scheduler_f],label_embeddings, args,logger)
        else:
            do_train(epoch,train_loader,backbone,quantizer,[optimizer_f,optimizer_Q],[scheduler_f,scheduler_Q],label_embeddings, args,logger)
        if epoch % val_epoch == 0:

            logger.info(f'test single-domain = ')
            test_mAP(args,S_query_set, S_gallery_set,True,True,backbone,top_k=top_k, dim=dim, logger=logger)
            
            logger.info(f'test cross-domain = ')
            mAP = test_mAP(args,T_query_set, S_gallery_set,False,True, backbone,top_k=top_k, dim=dim, logger=logger)


            logger.info(f'test single-domain quantized = ')
            test_mAP(args,S_query_set,S_gallery_set,True,True,backbone,Q=quantizer,top_k=top_k,dim=dim, logger=logger)
            
            logger.info(f'test cross-domain quantized = ')
            mAP = test_mAP(args,T_query_set,S_gallery_set,False,True,backbone,Q=quantizer,top_k=top_k,dim=dim, logger=logger)

            if mAP > best_map:
                best_map = mAP
                param_dict = {
                            'epoch':epoch, 
                            'backbone_state_dict':backbone.state_dict(),
                            'quantizer_state_dict':quantizer.state_dict(), 
                            'best_map':best_map,
                            }
                checkpoint_file = os.path.join(directory, save_name+'_'+str(epoch)+'.pth')
                torch.save(param_dict, checkpoint_file)
                logger.info('save at'+checkpoint_file)
    logger.info(f'best_mAP = {best_map:.3f}')