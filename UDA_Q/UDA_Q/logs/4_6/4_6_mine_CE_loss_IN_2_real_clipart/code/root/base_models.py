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

        feat = Intra_Norm(feat)
        return feat,predict