from audioop import avg
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
# from utils.logger import setup_logger


from quantizers.prob_quantizer import Prob_Quantizer

import logging


from fixed_dataset import split_dataset,pair_train_collate_fn,split_DomainNet

from my_dataset import get_CIFAR10, get_ImageNet

from evaluate import test_mAP

from processor.processor_uda import compute_knn_idx,generate_new_dataset,train_collate_fn,update_feat
from tool import Avg_er

import logging

from quantizers.Kmeans_PQ import Kmeans_PQ

from base_models import model_fc, model_vgg,model_AlexNet_single_head
from my_train import train_source,train_T_sample

from pseudo_label import NN_pseudo_label,prob_pseudo_label

def set_logger(file_path):
    logger = logging.getLogger('UDA_Q_log')
    logger.setLevel(level = logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s',"%Y-%m-%d-%H:%M:%S")

    file_handler = logging.FileHandler(file_path,mode='w')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


if __name__ == '__main__':
    args = Options().parse()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger = set_logger(args.log_dir)
    logger.info('Parameters:\t' + str(args))

    dim = 512

    if args.dataset == 'DomainNet':
        dataset_dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/domainnet/'
        S_train_set, S_query_set, S_gallery_set = split_DomainNet(args.seen_domain)
        T_train_set, T_query_set, T_gallery_set = split_DomainNet(args.unseen_domain)
        # S_train_set, S_query_set, S_gallery_set = split_dataset( dataset_dir+args.seen_domain+'/', False)
        # T_train_set, T_query_set, T_gallery_set = split_dataset( dataset_dir+args.unseen_domain+'/', False)
        n_class = 345
        top_k = 100
        backbone = model_fc(dim,n_class).cuda()
    elif args.dataset == 'CIFAR10':
        n_class = 10
        backbone = model_AlexNet_single_head(dim,n_class).cuda()
        S_train_set, S_query_set, S_gallery_set = get_CIFAR10()
        T_train_set, T_query_set, T_gallery_set = S_train_set, S_query_set, S_gallery_set
        top_k = -1
    elif args.dataset == 'ImageNet':
        n_class = 100
        backbone = model_AlexNet_single_head(dim,n_class).cuda()
        S_train_set, S_query_set, S_gallery_set = get_ImageNet()
        T_train_set, T_query_set, T_gallery_set = S_train_set, S_query_set, S_gallery_set
        top_k = 5000


    ic(S_train_set.__len__())
    ic(S_query_set.__len__())
    ic(S_gallery_set.__len__())


    ic(T_train_set.__len__())
    ic(T_query_set.__len__())
    ic(T_gallery_set.__len__())


    n_train = args.epochs
    batch_size = args.batch_size
    S_train_loader = DataLoader(S_train_set, batch_size=batch_size,
                            shuffle=True,  num_workers=4, pin_memory=True,)

    if args.dataset == 'DomainNet':
        optimizer = optim.SGD(backbone.parameters(), lr=2e-2, momentum=0.9, nesterov=True)
    elif args.dataset in [ 'CIFAR10','ImageNet' ] :
        # optimizer = optim.Adam(backbone.parameters(), lr=args.lr)
        optimizer = optim.SGD(backbone.get_optim_lr(args.lr), momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode= 'min',factor=0.3,min_lr=1e-3)


    train_loader1 = DataLoader( S_train_set, batch_size=batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=True)

    train_loader2 = DataLoader( T_train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,  pin_memory=True)

    core = np.load('core.npy')
    core = torch.FloatTensor(core).cuda()*10

    directory = args.model_dir
    save_name = '_'.join( (args.day , args.id) )

    best_map = -1


    quantizer = Prob_Quantizer(args.num_quantzer,args.n_codeword,512)
    quantizer = quantizer.cuda()

    Q_optimizer = optim.Adam(quantizer.parameters(), lr=args.Q_lr)
    # Q_optimizer = optim.SGD(quantizer.parameters(), lr=2e-2, momentum=0.9, nesterov=True)
    Q_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Q_optimizer,mode= 'min',factor=0.3,min_lr=1e-3)
    
    val_epoch = 20
    update_epoch = 10
    epoch_warmup = args.epoch_warmup
    
    threshold = args.threshold

    for epoch in range(n_train):
        if ( epoch % update_epoch == 0) and (epoch>=epoch_warmup ):

            # train_set = NN_pseudo_label(S_train_set,T_train_set,backbone,quantizer,logger)
            train_set = prob_pseudo_label(S_train_set,T_train_set,backbone,quantizer,threshold,logger)

            train_loader = DataLoader( train_set, batch_size=batch_size,shuffle=True,num_workers=4, pin_memory=True)

            # train_loader = DataLoader( train_set, batch_size=batch_size,shuffle=True,num_workers=4, pin_memory=True,
            # collate_fn=pair_train_collate_fn,)
            logger.info('\n')

        # print('epoch = ',epoch)
        logger.info(f'epoch = {epoch:d}')
        # train_source( epoch , core ,train_loader1, backbone, optimizer )
        
        # print(quantizer.CodeBooks[0])
        mean_loss = train_source( train_loader1, backbone, [optimizer,Q_optimizer],quantizer,logger )
        # mean_loss = 0

        if epoch >= epoch_warmup:
            # mean_loss += train_T_sample( epoch , core ,train_loader,S_train_set, backbone, [optimizer,Q_optimizer] , 
            mean_loss += train_T_sample( train_loader,S_train_set, backbone, [optimizer], 
            quantizer,logger )
            # mean_loss = train_S_T_pair( epoch , core ,train_loader,T_train_set, backbone, [optimizer,Q_optimizer] , quantizer,logger )

        scheduler.step(mean_loss)
        Q_scheduler.step(mean_loss)
        
        if epoch % val_epoch == 0:

            if args.dataset == 'DomainNet':
                logger.info(f'test single-domain = ')
                test_mAP(S_query_set, S_gallery_set, backbone,top_k=top_k, dim=dim, logger=logger)
            
            logger.info(f'test cross-domain = ')
            test_mAP(T_query_set, S_gallery_set, backbone,top_k=top_k, dim=dim, logger=logger)


            if args.dataset == 'DomainNet':
                logger.info(f'test single-domain quantized = ')
                test_mAP(S_query_set,S_gallery_set,backbone,Q=quantizer,top_k=top_k,dim=dim, logger=logger)

            
            logger.info(f'test cross-domain quantized = ')
            mAP = test_mAP(T_query_set,S_gallery_set,backbone,Q=quantizer,top_k=top_k,dim=dim, logger=logger)

            if mAP > best_map:
                best_map = mAP
                param_dict = {
                            'epoch':epoch, 
                            'backbone_state_dict':backbone.state_dict(),
                            'best_map':best_map,
                            }
                if quantizer is not None:
                    param_dict['quantizer_state_dict']=quantizer.state_dict()

                checkpoint_file = os.path.join(directory, save_name+'_'+str(epoch)+'.pth')
                torch.save(param_dict, checkpoint_file)
                # print('save at ',checkpoint_file)
                logger.info('save at '+checkpoint_file)
    logger.info(f'best_map = {best_map:.3f}')