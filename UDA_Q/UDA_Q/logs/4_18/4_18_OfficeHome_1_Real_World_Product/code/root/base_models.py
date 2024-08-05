from sympy import source
from torchvision import models
import torch.nn as nn
import torch
from tool import Intra_Norm

class model_vgg(nn.Module):
    def __init__(self):
        super(model_vgg, self).__init__()
        model = models.vgg16(pretrained=True)
        # print(model)

        self.features = model.features
        self.avgpool = model.avgpool

        self.mid = model.classifier[:6]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mid(x)
        return x

class model_fc(nn.Module):
    def __init__(self,dim,n_class):
        super(model_fc, self).__init__()
        self.n_class = n_class
        
        self.bn_fc = nn.BatchNorm1d(4096)
        self.fc = nn.Linear(4096, 4096)
        nn.init.xavier_uniform_(self.fc.weight)

        self.mid_cls = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.classifier = nn.Linear(4096, n_class)
        nn.init.xavier_uniform_(self.classifier.weight)

        
        self.mid_ext = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.extractor = nn.Linear(4096, dim)
        nn.init.xavier_uniform_(self.extractor.weight)

        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        base_f = self.fc(self.bn_fc(x))

        feat = self.tanh( self.extractor( self.mid_ext( base_f ) ) )
        
        predict = self.classifier( self.mid_cls( base_f ) )

        return feat,predict


class model_vgg_single_head(nn.Module):
    def __init__(self,dim,n_class):
        super(model_vgg_single_head, self).__init__()
        model = models.vgg16(pretrained=True)
        self.n_class = n_class

        self.features = model.features
        self.avgpool = model.avgpool

        self.mid = model.classifier[:6]

        self.bn_fc = nn.BatchNorm1d(4096)
        self.fc = nn.Linear(4096, 4096)
        nn.init.xavier_uniform_(self.fc.weight)

        self.mid_ext = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.extractor = nn.Linear(4096, dim)
        nn.init.xavier_uniform_(self.extractor.weight)

        self.mid_cls = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.classifier = nn.Linear(4096, n_class)
        nn.init.xavier_uniform_(self.classifier.weight)

        self.tanh = torch.nn.Tanh()

        self.base_f_ext = nn.Sequential( self.bn_fc , self.fc )
        self.extrator_branch = nn.Sequential( self.mid_ext , self.extractor,self.tanh)
        self.cls_branch = nn.Sequential( self.mid_cls , self.classifier)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mid(x)

        # base_f = self.fc(self.bn_fc(x))
        base_f = self.base_f_ext(x)
        feat = self.extrator_branch( base_f)
        predict = self.cls_branch( base_f)
        # feat = self.tanh( self.extractor( self.mid_ext( base_f ) ) )
        # predict = self.classifier( self.mid_cls( base_f ) )
        return feat,predict
    
    def get_optim_lr(self,base_lr):
        params = []
        params += [
                    {
                        "params": list(self.features.parameters())+list(self.mid.parameters()),
                        "lr": 1e-5,
                    }
                ]

        params += [
                    {
                        "params": list(self.base_f_ext.parameters())+list(self.extrator_branch.parameters())+list(self.cls_branch.parameters()),
                        "lr": base_lr,
                    }
                ]
        return params


class model_AlexNet_single_head(nn.Module):
    def __init__(self,dim,n_class):
        super(model_AlexNet_single_head, self).__init__()
        model = models.alexnet(pretrained=True)
        self.n_class = n_class

        self.features = model.features
        self.avgpool = model.avgpool

        self.mid = model.classifier[:6]

        self.bn_fc = nn.BatchNorm1d(4096)
        self.fc = nn.Linear(4096, 4096)
        nn.init.xavier_uniform_(self.fc.weight)

        self.mid_ext = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.extractor = nn.Linear(4096, dim)
        nn.init.xavier_uniform_(self.extractor.weight)

        self.mid_cls = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.classifier = nn.Linear(4096, n_class)
        nn.init.xavier_uniform_(self.classifier.weight)

        self.tanh = torch.nn.Tanh()

        self.base_f_ext = nn.Sequential( self.bn_fc , self.fc )
        self.extrator_branch = nn.Sequential( self.mid_ext , self.extractor,self.tanh)
        self.cls_branch = nn.Sequential( self.mid_cls , self.classifier)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mid(x)

        base_f = self.base_f_ext(x)
        feat = self.extrator_branch( base_f)
        predict = self.cls_branch( base_f)
        return feat,predict
    
    def get_optim_lr(self,base_lr):
        params = []
        params += [
                    {
                        "params": list(self.features.parameters())+list(self.mid.parameters()),
                        "lr": 1e-4,
                    }
                ]

        params += [
                    {
                        "params": list(self.base_f_ext.parameters())+list(self.extrator_branch.parameters())+list(self.cls_branch.parameters()),
                        "lr": base_lr,
                    }
                ]
        return params

class model_S_T_fc(nn.Module):
    def __init__(self,dim,n_class):
        super(model_S_T_fc, self).__init__()
        self.n_class = n_class
        self.dim = dim
        
        self.S_bn_fc = nn.BatchNorm1d(4096)
        self.S_fc = nn.Linear(4096, 4096)
        nn.init.xavier_uniform_(self.S_fc.weight)

        self.T_bn_fc = nn.BatchNorm1d(4096)
        self.T_fc = nn.Linear(4096, 4096)
        nn.init.xavier_uniform_(self.T_fc.weight)

        self.mid_cls = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.classifier = nn.Linear(4096, n_class)
        nn.init.xavier_uniform_(self.classifier.weight)

        
        self.mid_ext = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.extractor = nn.Linear(4096, dim)
        nn.init.xavier_uniform_(self.extractor.weight)


        # self.T_mid_ext = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        # self.T_extractor = nn.Linear(4096, dim)
        # nn.init.xavier_uniform_(self.T_extractor.weight)

        self.mid_ext = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.extractor = nn.Linear(4096, dim)
        nn.init.xavier_uniform_(self.extractor.weight)

        self.tanh = torch.nn.Tanh()

        self.S_base_f_ext = nn.Sequential( self.S_bn_fc , self.S_fc )
        self.T_base_f_ext = nn.Sequential( self.T_bn_fc , self.T_fc )

        self.extrator_branch = nn.Sequential( self.mid_ext , self.extractor,self.tanh)
        self.cls_branch = nn.Sequential( self.mid_cls , self.classifier)

    def forward(self, x , source_flag ):
        if source_flag == 'Source':
            base_f = self.S_base_f_ext(x)
        elif source_flag == 'Target':
            base_f = self.T_base_f_ext(x)

        feat = self.extrator_branch(base_f)
        predict = self.classifier( self.mid_cls( base_f ) )
        return feat,predict
