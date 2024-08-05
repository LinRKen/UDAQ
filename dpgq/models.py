from copy import deepcopy
from turtle import forward
from cv2 import normalize
# from numpy import zeros_like
from pytest import param
from torchvision import models
import torch.nn as nn
import torch
from icecream import ic


import sys
sys.path.append('..')
from base_models import model_vgg


# class Backbone_vgg(nn.Module):
#     def __init__(self, n_class, dim=300):
#         super(Backbone_vgg, self).__init__()

#         self.mid = nn.Sequential( nn.Linear(4096,4096) ,nn.LeakyReLU(),nn.Dropout(p=0.5) )
#         torch.nn.init.normal_(self.mid[0].weight.data, std=1e-2)

#         self.extractor = nn.Linear(4096, dim)
#         torch.nn.init.normal_(self.extractor.weight.data, std=1e-2)
#         torch.nn.init.constant_(self.extractor.bias.data, 0.0)

#         self.tanh = nn.Tanh()
#         self.classifier = nn.Linear(4096, n_class)
#         torch.nn.init.normal_(self.classifier.weight.data, std=1e-2)
#         torch.nn.init.constant_(self.classifier.bias.data, 0.0)

#     def forward(self, x):
#         # x = self.features(x)
#         x = self.mid(x)
#         out_predict = self.classifier(x)
#         out_feature = self.tanh(self.extractor(x))
#         return out_predict,out_feature

#     def param(self, lr):
#         params = []
#         # params += [{ "params": self.features.parameters(),"lr": lr,} ]
#         params += [{"params": self.extractor.parameters(),"lr": lr,}]
#         params += [{"params": self.classifier.parameters(),"lr": lr,}]
#         return params


class Quantizer_DPgQ(nn.Module):
    def __init__(self, n_quantizer, n_codeword, dim):
        super(Quantizer_DPgQ, self).__init__()

        self.CodeBooks = torch.nn.Parameter(torch.rand(n_quantizer,n_codeword, dim))
        torch.nn.init.normal_(self.CodeBooks.data, std=1e-2)
        self.n_quantizer = n_quantizer

    def forward(self, x: torch.FloatTensor):
        normalize = torch.nn.functional.normalize
        hard_x = torch.zeros_like(x)
        res = x.detach()
        
        MAX_id = torch.LongTensor(self.n_quantizer,x.size(0) ).cuda()+1000000
        
        for deep in range(self.n_quantizer):
            Codebook = self.CodeBooks[deep]
            dist = -torch.cdist(normalize(res), normalize(Codebook), p=2)

            MAX_id[deep] = torch.argmax(dist, 1)
            Q_hard = torch.index_select(Codebook, 0, MAX_id[deep])
            hard_x += Q_hard
            res -= Q_hard

        return hard_x,MAX_id.t()

    def Q_loss(self, x: torch.FloatTensor):
        soft_x = torch.zeros_like(x)
        hard_x = torch.zeros_like(x)
        softmax = torch.nn.Softmax(dim=1)
        normalize = torch.nn.functional.normalize
        # torch.autograd.set_detect_anomaly(True)

        res = torch.zeros_like(x)
        res.data += x
        res.requires_grad_(False)
        
        hard_loss = 0
        soft_loss = 0

        for deep in range(self.n_quantizer):
            a = normalize( res ).detach()
            b = normalize( self.CodeBooks[deep]).detach()
            
            dist = torch.mm(a, b.t())

            soft_x += torch.mm(softmax(5*dist) , self.CodeBooks[deep] )

            MAX_id = torch.argmax(dist, 1)
            Q_hard = torch.index_select(self.CodeBooks[deep], 0, MAX_id)
            hard_x += Q_hard

            res -= Q_hard

            hard_loss += torch.norm(x-hard_x, dim=1, p=2).mean()
            soft_loss += torch.norm(x-soft_x, dim=1, p=2).mean()
        joint_loss = torch.norm(soft_x - hard_x, dim=1, p=2).mean()
        hard_loss /= self.n_quantizer
        soft_loss /= self.n_quantizer
        return hard_loss , soft_loss ,joint_loss
