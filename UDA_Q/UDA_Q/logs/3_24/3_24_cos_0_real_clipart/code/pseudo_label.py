from cProfile import label
from sympy import appellf1
import torch
import time
from tqdm import tqdm
dim = 512


class list_dataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        super(list_dataset, self).__init__()
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


def generate_feature(base_f, backbone):
    f = torch.FloatTensor(base_f.size(0), dim).cuda()

    bs = 64
    cnt = 0
    with torch.no_grad():
        for st in range(0, base_f.size(0), bs):
            ed = min(st + bs, base_f.size(0))
            batch_base_f = base_f[st:ed]
            _, f[st:ed] = backbone(batch_base_f)
            cnt += (ed-st)
    assert(cnt == base_f.size(0))
    return f


def JS_dist(S_f, T_f, quantizer):
    sub_codebook = quantizer.CodeBooks[0]
    p_T_f = torch.softmax(torch.cdist(T_f[:, :128], sub_codebook), dim=1)
    p_S_f = torch.softmax(torch.cdist(S_f[:, :128], sub_codebook), dim=1)

    kl_loss = torch.nn.KLDivLoss(reduction="none")
    kl_dist = torch.zeros(S_f.size(0), T_f.size(0)).cuda()

    for i in range(S_f.size(0)):
        mid = (p_S_f[i].view(1, -1) + p_T_f)/2

        tmp = kl_loss(mid.log(), p_S_f[i].view(
            1, -1)).sum(dim=1) + kl_loss(mid.log(), p_T_f).sum(dim=1)
        kl_dist[i] = tmp
    return kl_dist

def get_S_T_pair(S_f,T_f,S_label,quantizer,metric):
    bs = 64
    T_idx = torch.LongTensor( S_f.size(0) )
    S_idx = torch.LongTensor( range(S_f.size(0) ) )
    for st in range(0, S_f.size(0), bs):
        ed = min(st+bs, S_f.size(0))
        batch_S_f = S_f[st:ed]
        batch_S_label = S_label[st:ed]

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



def NN_pseudo_label(S_train_set, T_train_set, backbone, quantizer=None):
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

    with torch.no_grad():
        S_idx,T_idx = get_S_T_pair(S_f,T_f,S_label,quantizer,'cos')

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
    print(
        f'\n\n pseudo_label acc = {acc:.3f} contain_rate = {contain_rate:.3f} ')
    time_elapsed = time.time() - st_time
    print('pseudo label complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return list_dataset(data_list)
