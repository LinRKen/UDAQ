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

class model_vgg_MNIST(nn.Module):
    def __init__(self):
        super(model_vgg_MNIST, self).__init__()
        model = models.vgg16(pretrained=True)
        # print(model)

        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class model_vgg_bn(nn.Module):
    def __init__(self):
        super(model_vgg_bn, self).__init__()
        model = models.vgg16_bn(pretrained=True)
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

class model_vgg_bn_all(nn.Module):
    def __init__(self,dim,n_class):
        super(model_vgg_bn_all, self).__init__()
        model = models.vgg16_bn(pretrained=True)
        # print(model)
        self.n_class = n_class

        self.features = model.features
        self.avgpool = model.avgpool
        self.vgg_mid = model.classifier[:6]

        # for x in list(self.features.parameters())+list(self.vgg_mid.parameters()):
        #     x.requires_grad_(False)

        hidden_dim = 4096
        self.fc = nn.Linear(4096, hidden_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias.data, 0.1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        

        self.mid = nn.Sequential( self.fc , self.bn, nn.LeakyReLU(),nn.Dropout(p=0.5))

        self.classifier = nn.Linear(hidden_dim, n_class)
        nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.constant_(self.classifier.bias.data, 0.1)

        self.extractor_fc = nn.Linear(hidden_dim, dim)
        nn.init.xavier_uniform_(self.extractor_fc.weight)
        torch.nn.init.constant_(self.extractor_fc.bias.data, 0.1)
        
        self.extractor = nn.Sequential( self.extractor_fc, nn.Tanh() )

    def get_optim_lr(self,base_lr):
        params = []
        params += [
                    {
                        "params": list(self.features.parameters())+list(self.vgg_mid.parameters()),
                        "lr": 1e-5,
                    }
                ]

        params += [
                    {
                        "params": list(self.extractor.parameters())+list(self.classifier.parameters()),
                        "lr": base_lr,
                    }
                ]
        return params

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.vgg_mid(x)
        base_f = self.mid(x)

        feat = self.extractor( base_f )
        predict = self.classifier( base_f )
        return feat,predict

class model_fc(nn.Module):
    def __init__(self,dim,n_class):
        super(model_fc, self).__init__()
        self.n_class = n_class
        
        self.bn_fc = nn.BatchNorm1d(4096)
        self.fc = nn.Linear(4096, 4096)
        nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias.data, 0.1)

        self.mid_cls = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.classifier = nn.Linear(4096, n_class)
        nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.constant_(self.classifier.bias.data, 0.1)

        
        self.mid_ext = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.extractor = nn.Linear(4096, dim)
        nn.init.xavier_uniform_(self.extractor.weight)
        torch.nn.init.constant_(self.extractor.bias.data, 0.1)

        self.tanh = torch.nn.Tanh()
        self.Is_source = True


    def forward(self, x):
        base_f = self.fc(self.bn_fc(x))

        feat = self.tanh( self.extractor( self.mid_ext( base_f ) ) )
        
        predict = self.classifier( self.mid_cls( base_f ) )

        return feat,predict


class model_fc_MNIST(nn.Module):
    def __init__(self,dim,n_class):
        super(model_fc_MNIST, self).__init__()
        self.n_class = n_class

        self.bn_fc = nn.BatchNorm1d(25088)

        self.fc = nn.Linear(25088, 4096)
        nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias.data, 0.1)


        self.mid_cls = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.classifier = nn.Linear(4096, n_class)
        nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.constant_(self.classifier.bias.data, 0.1)

        
        self.mid_ext = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.extractor = nn.Linear(4096, dim)
        nn.init.xavier_uniform_(self.extractor.weight)
        torch.nn.init.constant_(self.extractor.bias.data, 0.1)

        self.tanh = torch.nn.Tanh()
        self.Is_source = True


    def forward(self, x):
        base_f = self.fc( self.bn_fc(x) )

        feat = self.tanh( self.extractor( self.mid_ext( base_f ) ) )
        
        predict = self.classifier( self.mid_cls( base_f ) )

        return feat,predict

class model_refix_fc(nn.Module):
    def __init__(self,dim,n_class):
        super(model_refix_fc, self).__init__()
        self.n_class = n_class
        
        hidden_dim = 4096
        self.fc = nn.Linear(4096, hidden_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias.data, 0.1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        

        self.mid = nn.Sequential( self.fc , self.bn, nn.LeakyReLU(),nn.Dropout(p=0.5))

        self.classifier = nn.Linear(hidden_dim, n_class)
        nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.constant_(self.classifier.bias.data, 0.1)

        self.extractor_fc = nn.Linear(hidden_dim, dim)
        nn.init.xavier_uniform_(self.extractor_fc.weight)
        torch.nn.init.constant_(self.extractor_fc.bias.data, 0.1)
        
        self.extractor = nn.Sequential( self.extractor_fc , nn.Tanh() )
        self.Is_source = True


    def forward(self, x):
        base_f = self.mid(x)

        feat = self.extractor( base_f )
        
        predict = self.classifier( base_f )

        return feat,predict

class model_simple_fc(nn.Module):
    def __init__(self,dim):
        super(model_simple_fc, self).__init__()

        self.bn_fc = nn.BatchNorm1d(4096)
        self.fc = nn.Linear(4096, dim)
        nn.init.xavier_uniform_(self.fc.weight)

        self.extractor = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5),self.fc,nn.Tanh())
        self.Is_source = True

    def forward(self, x):
        feat = self.extractor( self.bn_fc(x) )
        return feat,feat

class model_fc_cls(nn.Module):
    def __init__(self,dim,n_class):
        super(model_fc_cls, self).__init__()

        self.bn_fc = nn.BatchNorm1d(4096)
        self.fc = nn.Linear(4096, dim)
        nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias.data, 0.1)

        self.extractor = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5),self.fc,nn.Tanh())

        self.classifier = nn.Linear(dim, n_class)
        torch.nn.init.constant_(self.classifier.bias.data, 0.1)
        nn.init.xavier_uniform_(self.classifier.weight)

        self.Is_source = True

    def forward(self, x):
        feat = self.extractor( self.bn_fc(x) )
        predict = self.classifier(feat)
        return feat,predict

class model_squeeze_fc(nn.Module):
    def __init__(self,dim,n_class):
        super(model_squeeze_fc, self).__init__()
        self.n_class = n_class
        
        self.bn_fc = nn.BatchNorm1d(4096)
        self.fc = nn.Linear(4096, 512)
        nn.init.xavier_uniform_(self.fc.weight)

        self.mid_cls = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.classifier = nn.Linear(512, n_class)
        nn.init.xavier_uniform_(self.classifier.weight)

        
        self.mid_ext = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.extractor = nn.Linear(512, dim)
        nn.init.xavier_uniform_(self.extractor.weight)

        self.tanh = torch.nn.Tanh()
        self.Is_source = True


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
        self.Is_source = True


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mid(x)

        # base_f = self.fc(self.bn_fc(x))
        base_f = self.base_f_ext(x)
        feat = self.extrator_branch( base_f)
        predict = self.cls_branch( base_f)
        
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

        self.Is_source = True


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
        
        self.bn_fc = nn.BatchNorm1d(4096)
        self.fc = nn.Linear(4096, 4096)
        nn.init.xavier_uniform_(self.fc.weight)

        self.mid_cls = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.classifier = nn.Linear(4096, n_class)
        nn.init.xavier_uniform_(self.classifier.weight)

        self.S_mid_ext = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.S_extractor = nn.Linear(4096, dim)
        nn.init.xavier_uniform_(self.S_extractor.weight)

        self.T_mid_ext = nn.Sequential( nn.LeakyReLU(),nn.Dropout(p=0.5))
        self.T_extractor = nn.Linear(4096, dim)
        nn.init.xavier_uniform_(self.T_extractor.weight)

        self.tanh = torch.nn.Tanh()

        self.base_f_ext = nn.Sequential( self.bn_fc , self.fc )

        self.S_extrator_branch = nn.Sequential( self.S_mid_ext , self.S_extractor,self.tanh)
        self.T_extrator_branch = nn.Sequential( self.T_mid_ext , self.T_extractor,self.tanh)

        self.cls_branch = nn.Sequential( self.mid_cls , self.classifier)

        self.Is_source = True

    def forward(self, x ):
        base_f = self.base_f_ext(x)

        if self.Is_source:
            feat = self.S_extrator_branch(base_f)
        else:
            feat = self.T_extrator_branch(base_f)

        predict = self.classifier( self.mid_cls( base_f ) )
        return feat,predict
