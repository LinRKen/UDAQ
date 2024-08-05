from asyncio.log import logger
from cProfile import label
from re import A
from numpy import dtype
from sqlalchemy import desc
from sympy import appellf1
import torch
import time
from tqdm import tqdm

from tool import Avg_er
dim = 512

import sys
sys.path.append('..')

from quantizers.prob_quantizer import Prob_Quantizer

class list_dataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        super(list_dataset, self).__init__()
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


def generate_feature(base_f, backbone):
    backbone.eval()
    f = torch.FloatTensor(base_f.size(0), dim).cuda()

    bs = 64
    cnt = 0
    with torch.no_grad():
        for st in range(0, base_f.size(0), bs):
            ed = min(st + bs, base_f.size(0))
            batch_base_f = base_f[st:ed]
            f[st:ed],_ = backbone(batch_base_f)
            cnt += (ed-st)
    assert(cnt == base_f.size(0))
    return f


def JS_dist(S_f, T_f, quantizer):
    sub_codebook = quantizer.CodeBooks[0]
    p_T_f = torch.softmax(torch.cdist(T_f[:, :128], sub_codebook), dim=1)
    p_S_f = torch.softmax(torch.cdist(S_f[:, :128], sub_codebook), dim=1)

    KL_loss = torch.nn.KLDivLoss(reduction="none")
    JS_dist = torch.zeros(S_f.size(0), T_f.size(0)).cuda()

    for i in range(S_f.size(0)):
        mid = (p_S_f[i].view(1, -1) + p_T_f)/2

        tmp = KL_loss(mid.log(), p_S_f[i].view(1, -1)).sum(dim=1) + KL_loss(mid.log(), p_T_f).sum(dim=1)
        JS_dist[i] = tmp
    return JS_dist

def get_S_T_pair(S_f,T_f,quantizer,metric):
    bs = 64
    T_idx = torch.LongTensor( S_f.size(0) )
    S_idx = torch.LongTensor( range(S_f.size(0) ) )
    with torch.no_grad():
        for st in range(0, S_f.size(0), bs):
            ed = min(st+bs, S_f.size(0))
            batch_S_f = S_f[st:ed]

            if metric == 'L2':
                dist = torch.cdist( batch_S_f , T_f)
            elif metric == 'mm':
                dist = -torch.mm( batch_S_f , T_f.T)
            elif metric == 'cos':
                dist = -torch.mm( torch.nn.functional.normalize(batch_S_f), torch.nn.functional.normalize((T_f)).T )
            elif metric == 'JS':
                dist = JS_dist(batch_S_f, T_f, quantizer)
            else:
                print('pseudo label metric error')
                assert( False)

            _, MIN_id = torch.min(dist, dim=1)

            T_idx[st:ed] = MIN_id

    return S_idx, T_idx

def get_prob_codebook(S_f : torch.FloatTensor,S_label:torch.LongTensor ,quantizer:Prob_Quantizer, n_class=345):
    n_quanizer = quantizer.n_quantizer
    n_codeword = quantizer.n_codeword
    prob = torch.zeros( n_quanizer, n_codeword, n_class ).cuda().view(-1)
    # std_prob = torch.zeros( n_quanizer, n_codeword, n_class ).cuda()
    
    base_M = torch.LongTensor( range(n_quanizer) ).cuda().view(1,-1)
    base_M *=n_codeword
    
    bs = 64
    
    cnt_cls = torch.zeros(345).cuda()
    with torch.no_grad():
        for st in range(0, S_f.size(0), bs):
            ed = min(st+bs, S_f.size(0))
            batch_S_f = S_f[st:ed]
            batch_S_label = S_label[st:ed]

            _ , MAX_id = quantizer(batch_S_f)

            cnt_cls += torch.bincount( batch_S_label,minlength=345)

            MAX_id_w_M = MAX_id + base_M
            MAX_id_w_M_label = MAX_id_w_M*n_class + batch_S_label.view(-1,1)
            MAX_id_w_M_label = MAX_id_w_M_label.reshape(-1)
            calc = torch.bincount(MAX_id_w_M_label , minlength= n_quanizer*n_codeword*n_class )
            prob += calc
    prob = prob.view(n_quanizer, n_codeword, n_class)
    prob_sum = torch.sum( prob , dim = 2).unsqueeze(-1) 
    prob /= (prob_sum+1e-7)
    return prob

def p_2_pseudo_label(T_f,quantizer,prob_code_cls,logger):
    n_f = T_f.size(0)
    n_quantizer = quantizer.n_quantizer
    n_cls = prob_code_cls.size(-1)
    prob_f_cls = torch.zeros( n_f , n_cls).cuda()

    mean_max_prob_f_code = Avg_er('mean_max_prob_f_code')
    
    bs = 128
    with torch.no_grad():
        for st in range(0, n_f, bs):
            ed = min(st+bs, n_f)
            batch_f = T_f[st:ed]
            batch_f = batch_f.view(batch_f.size(0),n_quantizer,-1)

            for deep in range(n_quantizer):
                dist = -torch.cdist( batch_f[ :, deep ,: ], quantizer.CodeBooks[deep] )
                prob_f_code = torch.nn.functional.softmax( 10*dist , dim = 1 )
                prob_f_cls[st:ed] += torch.mm( prob_f_code , prob_code_cls[deep] )/n_quantizer

                tmp,_ = torch.max(prob_f_code,dim=1)
                mean_max_prob_f_code.add( tmp.mean() , tmp.size(0))
    logger.info(mean_max_prob_f_code.out_s())

    tmp,pseudo_label = torch.max( prob_f_cls , dim = 1)
    tmp = tmp.mean()
    logger.info(f'prob_f_cls max = {tmp:.3f}')
    return pseudo_label , prob_f_cls


def prob_pseudo_label(S_train_set, T_train_set, backbone, quantizer,threshold,logger):
    backbone.eval()

    with torch.no_grad():
        S_f = generate_feature(S_train_set.f.cuda(), backbone)
        T_f = generate_feature(T_train_set.f.cuda(), backbone)

        S_label = torch.LongTensor(S_train_set.label).cuda()
        T_label = torch.LongTensor(T_train_set.label).cuda()

        prob_CB_cls = get_prob_codebook(S_f ,S_label ,quantizer, 345)
        pseudo_label , prob_f_cls = p_2_pseudo_label(T_f,quantizer,prob_CB_cls,logger)
        prob_pseudo_label, _ = torch.max(prob_f_cls,dim=1)
        

        sorted_idx = torch.argsort( prob_pseudo_label,descending=True)

        tmp = sorted_idx[0]
        print(tmp)
        logger.info(f' max prob = {prob_pseudo_label[ tmp ]:.3f}')
        for part_rate in [0.1,0.3,0.5]:
            part_idx = sorted_idx[ int(sorted_idx.size(0)*part_rate) ]
            logger.info(f'part_rate= {part_rate:.1f} part prob = {prob_pseudo_label[ part_idx ]:.3f}')
        
        idx = torch.LongTensor( range(T_f.size(0) ) ).cuda()

        if threshold > 0:
            idx = idx[ prob_pseudo_label >= threshold ]
        else:
            idx = sorted_idx[ : int(T_f.size(0)*0.3)]

        part_base_f = T_train_set.f[idx.cpu()]
        part_T_label = T_label[idx]
        pseudo_label = pseudo_label[idx]

        data_list = []
        for i in range(part_base_f.size(0)):
            data_list.append( ( part_base_f[i],pseudo_label[i].item()) )

        n_data = len(data_list)
        rate = n_data/T_f.size(0)
        acc = (pseudo_label == part_T_label).float().mean()
    
    logger.info(f'len_data = {n_data:d} contain_rate = {rate:.3f} pseudo_label acc = {acc:.3f}')
    return list_dataset(data_list)




def NN_pseudo_label(S_train_set, T_train_set, backbone, quantizer=None,logger=None):
    st_time = time.time()


    S_num = S_train_set.__len__()
    T_num = T_train_set.__len__()


    backbone.eval()

    S_f = generate_feature(S_train_set.f.cuda(), backbone)
    T_f = generate_feature(T_train_set.f.cuda(), backbone)

    S_label = torch.LongTensor(S_train_set.label)
    T_label = torch.LongTensor(T_train_set.label)

    data_list = []

    true_pair = torch.zeros(S_num).cuda()
    valid_sample = torch.zeros(T_num).cuda()

    p_code_cls = get_prob_codebook(S_f,S_label.cuda(),quantizer)
    MAX_p,_ = torch.max( p_code_cls, dim =2)
    logger.info(f'MAX_p_code_cls_mean = {MAX_p.mean():.3f} ')

    pseudo_label = p_2_pseudo_label( T_f , quantizer, p_code_cls,logger )

    acc = (pseudo_label.cpu() == T_label).float().mean()
    logger.info(f'p_2_pseudo_label acc = {acc:.3f}')
    
    with torch.no_grad():
        S_idx,T_idx = get_S_T_pair(S_f,T_f,quantizer,'L2')

        # if quantizer is not None:
        #     S_JS_idx, T_JS_idx = get_S_T_pair(S_f,T_f,S_label,quantizer,'JS')

        #     L2_label = torch.zeros(T_num).long()
        #     L2_label -= 1

        #     L2_label[ T_idx ] = S_label[ S_idx ]

        #     new_S_idx = []
        #     new_T_idx = []
            
        #     for i in range(S_JS_idx.size(0)):
        #         S_id = S_JS_idx[i]
        #         T_id = T_JS_idx[i]
        #         if ( L2_label[T_id] ==-1)  or ( S_label[S_id] == L2_label[T_id] ):
        #             new_S_idx.append( S_id )
        #             new_T_idx.append( T_id )
        #         else:
        #             new_S_idx.append( S_id )
        #             new_T_idx.append( T_idx[i] )
        #     S_idx = torch.LongTensor( new_S_idx )
        #     T_idx = torch.LongTensor( new_T_idx )

        pseudo_label = S_label[ S_idx ]
        true_label = T_label[ T_idx ]
        true_pair = (true_label == pseudo_label ).float()
        valid_sample[ T_idx ] = 1

        for i in range(S_idx.size(0)):
            S_id = S_idx[i]
            T_id = T_idx[i]
            label = S_label[S_id]
            pseudo_label = label
            data_list.append(
                (S_train_set.f[S_id].cpu(), T_train_set.f[T_id].cpu(), label, pseudo_label, S_id, T_id,))

    acc = true_pair.sum() / true_pair.size(0)
    contain_rate = valid_sample.sum() / valid_sample.size(0)
    # print(
    logger.info(
        f'\n\n pseudo_label acc = {acc:.3f} contain_rate = {contain_rate:.3f} ')
    time_elapsed = time.time() - st_time
    # print(f'pseudo label complete in {time_elapsed // 60:.0f}m { time_elapsed % 60:.0f}s')
    logger.info(f'pseudo label complete in {time_elapsed // 60:.0f}m { time_elapsed % 60:.0f}s')
    return list_dataset(data_list)
