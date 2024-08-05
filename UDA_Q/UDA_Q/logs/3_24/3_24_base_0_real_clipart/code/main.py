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
from utils.logger import setup_logger



from fixed_dataset import fixed_dataset,pair_dataset
from fixed_dataset import split_dataset,pair_train_collate_fn

from evaluate import test_mAP

from processor.processor_uda import compute_knn_idx,generate_new_dataset,train_collate_fn,update_feat
from tool import Avg_er

import logging

from quantizers.Kmeans_PQ import Kmeans_PQ

from base_models import model_fc, model_vgg
from my_train import train_source,train_S_T_pair

from pseudo_label import NN_pseudo_label

if __name__ == '__main__':
    args = Options().parse()

    print('Parameters:\t' + str(args))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/domainnet/'

    S_train_set, S_query_set, S_gallery_set = split_dataset( dataset_dir+args.seen_domain+'/', False)
    T_train_set, T_query_set, T_gallery_set = split_dataset( dataset_dir+args.unseen_domain+'/', False)

    # for domain in ['painting','quickdraw','real','sketch','infograph','clipart']:
    #     train_set ,query_set, gallery_set=split_dataset( 'fixed_f/vgg/domainnet/'+domain+'/', False)
    #     print('OK ',domain)
    #     ic(train_set.__len__())
    #     ic(query_set.__len__())
    #     ic(gallery_set.__len__())
    # exit()
    # gallery_set,_ = split_dataset('fixed_f/vgg/domainnet/real',True)

    ic(S_train_set.__len__())
    ic(S_query_set.__len__())
    ic(S_gallery_set.__len__())


    ic(T_train_set.__len__())
    ic(T_query_set.__len__())
    ic(T_gallery_set.__len__())

    # exit()

    dim = 512
    n_class = 345
    backbone = model_fc(dim,n_class).cuda()

    n_train = 100
    batch_size = 64
    S_train_loader = DataLoader(S_train_set, batch_size=batch_size,
                            shuffle=True,  num_workers=4, pin_memory=True,)

    optimizer = optim.SGD(backbone.parameters(), lr=1e-2, momentum=0.9, nesterov=True)


    train_loader1 = DataLoader( S_train_set, batch_size=batch_size, shuffle=True,num_workers=4, pin_memory=True)

    train_loader2 = DataLoader( T_train_set, batch_size=batch_size, shuffle=True, num_workers=4,  pin_memory=True)

    train_loader = 1

    n_source = S_train_set.__len__()
    n_target = T_train_set.__len__()

    label_memory1 = torch.zeros((n_source),dtype=torch.long)
    label_memory2 = torch.zeros((n_target),dtype=torch.long)

    feat_memory1 = torch.zeros((n_source,dim),dtype=torch.float32)
    feat_memory2 = torch.zeros((n_target,dim),dtype=torch.float32)

    device='cuda:0'

    class tmp_dataset():
        def __init__(self,list) -> None:
            self.train = list

    s_dataset = tmp_dataset( S_train_set.get_list() )
    t_dataset = tmp_dataset( T_train_set.get_list() )
    output_dir = './mine_log'

    logger = setup_logger("reid_baseline", output_dir, if_train=True)


    loss_CE = torch.nn.CrossEntropyLoss()


    core = np.load('core.npy')
    core = torch.FloatTensor(core).cuda()*10

    

    directory = args.model_dir
    save_name = '_'.join( (args.day , args.id) )

    best_map = -1

    from my_train import Quantizer

    quantizer = Quantizer(args.num_quantzer,args.n_codeword,512)
    quantizer = quantizer.cuda()

    Q_optimizer = optim.Adam(quantizer.parameters(), lr=args.Q_lr)
    
    val_epoch = 5
    update_epoch = 10
    epoch_warmup = 0

    for epoch in range(n_train):
        if ( epoch % update_epoch == 0) and (epoch>=epoch_warmup ):

            # train_set = NN_pseudo_label(S_train_set,T_train_set,backbone,quantizer)
            train_set = NN_pseudo_label(S_train_set,T_train_set,backbone)
            # feat_memory1, feat_memory2, label_memory1, label_memory2 = update_feat( None , epoch, backbone, train_loader1,train_loader2, device,feat_memory1,feat_memory2, label_memory1,label_memory2)

            # dynamic_top = 1
            # print('source and target topk==',dynamic_top)
            # logger = logging.getLogger("reid_baseline.train")
            # target_label, knnidx, knnidx_topk, target_knnidx = compute_knn_idx(logger, backbone, train_loader1, train_loader2, feat_memory1, feat_memory2, label_memory1, label_memory2, n_source, n_target, topk=dynamic_top, reliable_threshold=0.0)
            # del train_loader
            # train_data_list = generate_new_dataset(None, logger, label_memory2, s_dataset, t_dataset, knnidx, target_knnidx, target_label, label_memory1,
            # n_source, n_target, with_pseudo_label_filter = False,only_dataset_out=True,only_source=True)

            # train_set = pair_dataset(train_data_list)

            train_loader = DataLoader( train_set, batch_size=batch_size,shuffle=True,num_workers=4, pin_memory=True,
            collate_fn=pair_train_collate_fn,)

        print('epoch = ',epoch)
        # train_source( epoch , core ,train_loader1, backbone, optimizer )
        
        if epoch >= epoch_warmup:
            # train_source( epoch , core ,train_loader2, backbone, optimizer )
            train_S_T_pair( epoch , core ,train_loader,T_train_set, backbone, [optimizer,Q_optimizer] , quantizer=quantizer )
            # train_S_T_pair( epoch , core ,train_loader,T_train_set, backbone, [optimizer]  )
        else:
            train_source( epoch , core ,train_loader1, backbone, [optimizer,Q_optimizer],  quantizer=quantizer )
            # train_source( epoch , core ,train_loader1, backbone, [optimizer])
        
        if epoch % val_epoch == 0:
            print('test cross-domain ',end=' ')
            test_mAP(T_query_set, S_gallery_set, backbone, dim=dim)
            print('test single-domain ',end=' ')
            test_mAP(S_query_set, S_gallery_set, backbone, dim=dim)

            mAP = test_mAP(T_query_set,S_gallery_set,backbone,Q=quantizer,top_k=100,dim=dim)
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
                print('save at ',checkpoint_file)