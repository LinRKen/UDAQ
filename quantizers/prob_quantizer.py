import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
import sys

sys.path.append('..')
from tool import Intra_Norm

class Prob_Quantizer(nn.Module):
    def __init__(self, n_quantizer, n_codeword, dim):
        super(Prob_Quantizer, self).__init__()

        self.sub_dim = dim//n_quantizer
        assert( dim % n_quantizer == 0 )

        self.CodeBooks = torch.nn.Parameter(torch.rand(n_quantizer, n_codeword, self.sub_dim))
        self.n_quantizer = n_quantizer
        self.n_codeword = n_codeword
        self.prob_CB_cls = None
    
    def to_gpu(self):
        self.CodeBooks = self.CodeBooks.cuda()

    def tanh(self):
        with torch.no_grad():
            self.CodeBooks.data = torch.tanh(self.CodeBooks)

    def Intra_Normalize(self):
        with torch.no_grad():
            self.CodeBooks.data = torch.nn.functional.normalize(self.CodeBooks.data,dim=2)

    def forward(self, x: torch.FloatTensor, soft_rate=-1):
        normalize = torch.nn.functional.normalize
        split_x = x.detach()
        split_x = split_x.view( -1 , self.n_quantizer, self.sub_dim)
        Q_x = torch.zeros_like(split_x)
        
        MAX_id = torch.LongTensor(self.n_quantizer,x.size(0) ).cuda()+1000000

        for deep in range(self.n_quantizer):
            Codebook = self.CodeBooks[deep]
            part_x = split_x[:,deep,:]
            dist = -torch.cdist(part_x, Codebook, p=2)
            
            if soft_rate==-1:
                MAX_id[deep] = torch.argmax( dist , dim = 1)
                Q_x[:,deep,:] = torch.index_select(Codebook, 0, MAX_id[deep])
            else:
                soft_dist = F.softmax( dist*soft_rate , dim =1 )
                Q_x[:,deep,:] = torch.mm( soft_dist , Codebook )
        Q_x = Q_x.view(-1,x.size(1))

        return Q_x,MAX_id.t()
    
    def all_code(self,x:torch.FloatTensor):
        bs = 128
        Q_id = torch.zeros( x.size(0), self.n_quantizer).long().cuda()+1000000
        with torch.no_grad():
            for st in range(0, x.size(0), bs):
                ed = min(st+bs, x.size(0))
                batch_S_f = x[st:ed]

                _ , Q_id[st:ed] = self.forward(batch_S_f)
        return Q_id
    
    def get_prob_CB_cls(self,S_f : torch.FloatTensor,S_label:torch.LongTensor, n_class):
        n_quantizer = self.n_quantizer
        n_codeword = self.n_codeword
        prob = torch.zeros( n_quantizer, n_codeword, n_class ).cuda().view(-1)
        prob.requires_grad_(False)
        
        base_M = torch.LongTensor( range(n_quantizer) ).cuda().view(1,-1)
        base_M *=n_codeword
        
        bs = 64
        with torch.no_grad():
            for st in range(0, S_f.size(0), bs):
                ed = min(st+bs, S_f.size(0))
                batch_S_f = S_f[st:ed]
                batch_S_label = S_label[st:ed]

                _ , MAX_id = self.forward(batch_S_f)

                MAX_id_w_M = MAX_id + base_M
                MAX_id_w_M_label = MAX_id_w_M*n_class + batch_S_label.view(-1,1)
                MAX_id_w_M_label = MAX_id_w_M_label.reshape(-1)
                calc = torch.bincount(MAX_id_w_M_label , minlength= n_quantizer*n_codeword*n_class )
                prob += calc
        prob = prob.view(n_quantizer, n_codeword, n_class)
        prob_sum = torch.sum( prob , dim = 2).unsqueeze(-1) 
        prob /= (prob_sum+1e-7)
        return prob
    
    def update_prob_CB_cls(self,S_f : torch.FloatTensor,S_label:torch.LongTensor, n_class,gamma):
        prob_CB_cls = self.get_prob_CB_cls(S_f,S_label,n_class)
        if self.prob_CB_cls is None:
            self.prob_CB_cls = prob_CB_cls
        else:
            self.prob_CB_cls = self.prob_CB_cls*gamma + prob_CB_cls*(1-gamma)


    def predict_cls(self,feat,prob_CB_cls):

        n_f = feat.size(0)
        n_quantizer = self.n_quantizer
        n_cls = prob_CB_cls.size(-1)
        prob_f_cls = torch.zeros( n_f , n_cls).cuda()
        
        feat = feat.view(feat.size(0),n_quantizer,-1)

        for deep in range(n_quantizer):
            dist = -torch.cdist( feat[ :, deep ,: ], self.CodeBooks[deep] )
            prob_f_code = torch.nn.functional.softmax( 10*dist , dim = 1 )
            prob_f_cls += torch.mm( prob_f_code , prob_CB_cls[deep] )/n_quantizer
        return prob_f_cls