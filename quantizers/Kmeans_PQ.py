import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from base_Q import Base_Q

class Kmeans_PQ(nn.Module):
    def __init__(self,n_quantizer,n_codeword, len_vec):
        super(Kmeans_PQ,self).__init__()

        self.n_quantizer = n_quantizer
        self.n_codeword = n_codeword
        self.len_vec = len_vec
        self.len_subvec = len_vec // n_quantizer

        assert( self.len_subvec * n_quantizer == len_vec )
        self.list_Q = [ Base_Q(n_codeword,self.len_subvec).cuda() for _ in range(n_quantizer) ]
    
    def forward(self,x,soft_rate=-1):
        x = x.view( x.size(0) , self.n_quantizer  , self.len_subvec )
        Q_x = torch.zeros_like( x )
        id_x = torch.LongTensor( self.n_quantizer, x.size(0) )
        for deep in range(self.n_quantizer):
            Q_x[:,deep,:],id_x[deep] = self.list_Q[deep]( x[:,deep,:] , soft_rate )
        return Q_x.view(-1,self.len_vec),id_x


    def train(self,loader_image=None,model=None,all_im_em=None):
        st_time = time.time()
        if all_im_em is None:
            for i, data in enumerate(loader_image):
                im = data[0]
                im = im.float().cuda()
                with torch.no_grad():
                    _, im_feat = model(im)
                    im_em = model.base_model.last_linear(im_feat)
                    
                    if i == 0:
                        all_im_em = im_em.cpu().data.numpy()
                    else:
                        all_im_em = np.concatenate((all_im_em, im_em.cpu().data.numpy()), axis=0)
        print('train quantizer on data of',all_im_em.shape)
        from sklearn.cluster import KMeans,MiniBatchKMeans
        for deep in range(self.n_quantizer):
            sub_im_em = all_im_em[ : , deep*self.len_subvec : (deep+1)*self.len_subvec ]
            kmeans = MiniBatchKMeans(n_clusters=self.n_codeword ,tol=1e-3).fit(sub_im_em)
            centroids = kmeans.cluster_centers_
            self.list_Q[deep].CodeBook.data = torch.FloatTensor( centroids ).cuda()
        delta = time.time() - st_time
        print(f"Train Time:{delta//60:.0f}m{delta%60:.0f}s")