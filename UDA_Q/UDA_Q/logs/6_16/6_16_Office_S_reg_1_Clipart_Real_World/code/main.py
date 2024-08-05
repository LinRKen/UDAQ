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


from fixed_dataset import split_OfficeHome,split_DomainNet,get_MNIST_USPS

from my_dataset import get_CIFAR10, get_ImageNet, get_img_DomainNet,get_img_MNIST_USPS

from evaluate import test_mAP

from processor.processor_uda import compute_knn_idx,generate_new_dataset,train_collate_fn,update_feat
from tool import Avg_er

import logging

from quantizers.Kmeans_PQ import Kmeans_PQ

from base_models import model_fc, model_vgg,model_AlexNet_single_head,model_S_T_fc,model_vgg_single_head
from base_models import model_simple_fc,model_fc_cls,model_refix_fc,model_vgg_bn_all
from base_models import model_fc_MNIST,model_simple_fc_MNIST
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

    dim = args.dim
    logger.info('dim = '+str(dim))

    if args.dataset == 'DomainNet':
        S_train_set, S_query_set, S_gallery_set = split_DomainNet(args.seen_domain)
        T_train_set, T_query_set, T_gallery_set = split_DomainNet(args.unseen_domain)

        # S_train_set, S_query_set, S_gallery_set = get_img_DomainNet(args.seen_domain)
        # T_train_set, T_query_set, T_gallery_set = get_img_DomainNet(args.unseen_domain)
        args.n_class = 345
        top_k = -1
        # backbone = model_fc(dim,args.n_class).cuda()
        # backbone = model_fc_MNIST(dim,args.n_class).cuda()
        if args.epochs == 201:
            backbone = model_simple_fc_MNIST(dim,args.n_class).cuda()
            logger.info('model_simple_fc_MNIST')
            logger.info('==================')
        else:
            backbone = model_fc_MNIST(dim,args.n_class).cuda()
        
        # backbone = model_S_T_fc(dim,args.n_class).cuda()
        # backbone = model_vgg_single_head(dim,args.n_class).cuda()
    elif args.dataset == 'OfficeHome':
        _, S_query_set, S_train_set = split_OfficeHome(args.seen_domain)
        S_gallery_set = S_train_set

        T_train_set, T_query_set, _ = split_OfficeHome(args.unseen_domain)

        args.n_class = 65
        top_k = -1
        backbone = model_fc(dim,args.n_class).cuda()
        # backbone = model_fc_MNIST(dim,args.n_class).cuda()
    elif args.dataset == 'MNIST_USPS':
        args.n_class = 10
        top_k = -1
        S_train_set, S_query_set, T_train_set, T_query_set=get_MNIST_USPS()
        S_gallery_set = S_train_set
        # S_train_set, S_query_set, S_gallery_set, T_train_set, T_query_set, T_gallery_set=get_img_MNIST_USPS()
        # backbone = model_fc_cls(dim,args.n_class).cuda()
        # backbone = model_fc(dim,args.n_class).cuda()
        backbone = model_fc_MNIST(dim,args.n_class).cuda()
        # backbone = model_vgg_bn_all(dim,args.n_class).cuda()
        # backbone = model_refix_fc(dim,args.n_class).cuda()
        # backbone = model_simple_fc(dim).cuda()


    logger.info('Parameters:\t' + str(args))
    ic(S_train_set.__len__())
    ic(S_query_set.__len__())
    ic(S_gallery_set.__len__())


    ic(T_train_set.__len__())
    ic(T_query_set.__len__())


    n_train = args.epochs
    batch_size = args.batch_size
    S_train_loader = DataLoader(S_train_set, batch_size=batch_size,
                            shuffle=True,  num_workers=4, pin_memory=True,)
# model_fc, model_vgg,model_AlexNet_single_head,model_S_T_fc,model_vgg_single_head
    if args.optimizer == 'sgd':
        S_optimizer = optim.SGD(backbone.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'adam':
        S_optimizer = optim.Adam(backbone.parameters(), lr=args.lr,betas=(0.5,0.5),weight_decay=args.weight_decay)

    # T_optimizer = optim.SGD(backbone.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    T_optimizer = S_optimizer

    S_scheduler = torch.optim.lr_scheduler.ExponentialLR(S_optimizer,gamma=0.95,)
    # T_scheduler = torch.optim.lr_scheduler.ExponentialLR(T_optimizer,gamma=0.95,)
    # S_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(S_optimizer,mode= 'min',factor=0.3,min_lr=1e-3)
    # T_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(T_optimizer,mode= 'min',factor=0.3,min_lr=1e-3)


    S_train_loader = DataLoader( S_train_set, batch_size=batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=True)

    T_train_loader = DataLoader( T_train_set, batch_size=batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=True)

    # core = np.load('core.npy')
    # core = torch.FloatTensor(core).cuda()*10

    directory = args.model_dir
    save_name = '_'.join( (args.day , args.id) )

    best_map = -1


    quantizer = Prob_Quantizer(args.num_quantzer,args.n_codeword,dim)
    quantizer = quantizer.cuda()

    # Q_optimizer = optim.Adam(quantizer.parameters(), lr=args.Q_lr,betas=(0.5,0.99))
    Q_optimizer = optim.Adam(quantizer.parameters(), lr=args.Q_lr,weight_decay=args.Q_weight_decay)
    # Q_optimizer = optim.SGD(quantizer.parameters(), lr=args.Q_lr, momentum=0.9, nesterov=True)
    # Q_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Q_optimizer,mode= 'min',factor=0.3,min_lr=1e-3)
    Q_scheduler = torch.optim.lr_scheduler.ExponentialLR(Q_optimizer,gamma=0.99,)
    
    val_epoch = 10

    update_epoch = 10
    epoch_warmup = args.epoch_warmup
    
    threshold = args.threshold

    # if True:
    if False:
        st_epoch = 100
        load_model_path = '/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/logs/5_10/5_10_DomainNet_vgg_MNIST_7_real_clipart/model/5_10_DomainNet_vgg_MNIST_7_70.pth'
        logger.info('load from '+load_model_path)
        checkpoint = torch.load(load_model_path)
        backbone.load_state_dict( checkpoint['backbone_state_dict'] )
        backbone.eval()

        quantizer.load_state_dict( checkpoint['quantizer_state_dict'])

        logger.info(f'test single-domain quantized = ')
        test_mAP(args,S_query_set,S_gallery_set,True,True,backbone,Q=quantizer,top_k=top_k,dim=dim, logger=logger)

    
        logger.info(f'test cross-domain quantized = ')
        mAP = test_mAP(args,T_query_set,S_gallery_set,False,True,backbone,Q=quantizer,top_k=top_k,dim=dim, logger=logger)
        exit()

    else:
        st_epoch = 0


    for epoch in range(st_epoch,n_train+1):
        # if ( epoch % update_epoch == 0) and (epoch>=epoch_warmup ):
        logger.info('lr = '+str(S_optimizer.param_groups[-1]['lr']))
        if ( epoch % update_epoch == 0):

        #     # train_set = NN_pseudo_label(S_train_set,T_train_set,backbone,quantizer,logger)
            pseudo_train_set = prob_pseudo_label(S_train_loader,T_train_loader,backbone,quantizer,threshold,logger)

        #     pseudo_train_loader = DataLoader( pseudo_train_set, batch_size=batch_size,shuffle=True,num_workers=4, pin_memory=True)

        #     # train_loader = DataLoader( train_set, batch_size=batch_size,shuffle=True,num_workers=4, pin_memory=True,
        #     # collate_fn=pair_train_collate_fn,)
        #     logger.info('\n')

        # print('epoch = ',epoch)
        logger.info(f'epoch = {epoch:d}')
        # train_source( epoch , core ,S_train_loader, backbone, optimizer )
        
        # print(quantizer.CodeBooks[0])

        from pseudo_label import generate_feature
        
        with torch.no_grad():
            backbone.eval()
            S_f,S_label = generate_feature(S_train_loader,backbone)
            quantizer.update_prob_CB_cls(S_f,S_label,args.n_class,args.cls_gamma)
            prob_CB_cls = quantizer.prob_CB_cls


        S_loss = train_source(args, S_train_loader, backbone,prob_CB_cls, [S_optimizer,Q_optimizer],quantizer,logger)
        # mean_loss = 0

        if epoch >= epoch_warmup:
            # mean_loss += train_T_sample( epoch , core ,train_loader,S_train_set, backbone, [optimizer,Q_optimizer] , 
            # mean_loss += train_T_sample( pseudo_train_loader,S_train_loader, backbone, [optimizer], 
            # quantizer,logger )
            T_loss = train_T_sample(args, T_train_loader, backbone,prob_CB_cls,[T_optimizer], 
            quantizer,logger )
            # mean_loss += train_T_sample( T_train_loader,S_train_loader, backbone, [optimizer,Q_optimizer], 
            # quantizer,logger )
            # mean_loss = train_S_T_pair( epoch , core ,train_loader,T_train_set, backbone, [optimizer,Q_optimizer] , quantizer,logger )
            # if epoch%10==0:
            #     T_scheduler.step()

        # S_scheduler.step(S_loss)
        # Q_scheduler.step(S_loss)
        if epoch%10==0:
            S_scheduler.step()
            Q_scheduler.step()
        
        if epoch % val_epoch == 0:

            if args.dataset in ['DomainNet','OfficeHome','MNIST_USPS']:
                logger.info(f'test single-domain = ')
                test_mAP(args,S_query_set, S_gallery_set,True,True,backbone,top_k=top_k, dim=dim, logger=logger)
            
            logger.info(f'test cross-domain = ')
            test_mAP(args,T_query_set, S_gallery_set,False,True, backbone,top_k=top_k, dim=dim, logger=logger)


            if args.dataset  in ['DomainNet','OfficeHome','MNIST_USPS']:
                logger.info(f'test single-domain quantized = ')
                test_mAP(args,S_query_set,S_gallery_set,True,True,backbone,Q=quantizer,top_k=top_k,dim=dim, logger=logger)

            
            logger.info(f'test cross-domain quantized = ')
            mAP = test_mAP(args,T_query_set,S_gallery_set,False,True,backbone,Q=quantizer,top_k=top_k,dim=dim, logger=logger)

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