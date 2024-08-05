import torch
import numpy as np
import sys
from icecream import ic
import matplotlib.pyplot as plt
import random
sys.path.append('..')

from base_models import model_fc
from tqdm import tqdm

title_bias = -0.35

def tsne(see_label , S_f,S_label,T_f,T_label,):

    see_S_f = torch.zeros_like(S_f[:1])
    see_T_f = torch.zeros_like(S_f[:1])
    see_S_label = torch.zeros_like(S_label[:1])
    see_T_label = torch.zeros_like(S_label[:1])
    with torch.no_grad():
        for label in see_label:
            S_idx = torch.where( S_label == label)
            T_idx = torch.where( T_label == label)
            see_S_f = torch.cat( (see_S_f ,  S_f[ S_idx ]),dim = 0)
            see_T_f = torch.cat( (see_T_f ,  T_f[ T_idx ]),dim = 0)

            see_S_label = torch.cat( (see_S_label ,  S_label[ S_idx ]),dim = 0)
            see_T_label = torch.cat( (see_T_label ,  T_label[ T_idx ]),dim = 0)
        
        see_S_f = see_S_f[1:].detach().cpu()
        see_T_f = see_T_f[1:].detach().cpu()
        see_S_label = see_S_label[1:].detach().cpu()
        see_T_label = see_T_label[1:].detach().cpu()

    n_see_S_f = see_S_f.size(0)
    print('features load OK')
    print( see_S_f.size() , see_T_f.size() )
    from sklearn.manifold import TSNE
    import numpy as np
    n_perplexity = 50
    all_embed = torch.cat( (see_S_f,see_T_f) ,dim=0 )
    # embedded = TSNE(perplexity= n_perplexity, init='pca'  , method='exact' , learning_rate = 200 ).fit_transform( all_embed.data )
    embedded = TSNE(perplexity= n_perplexity ).fit_transform( all_embed.data )
    print('t-SNE OK')
    return embedded[:n_see_S_f], see_S_label ,embedded[n_see_S_f:], see_T_label

def plot_tsne(see_label,data,subfig,title):
    color = ['red','blue','green','gray','pink', 'gold','purple','orange','dodgerblue','cyan']
    S_f,S_label,T_f,T_label = data

    for i in range( len(see_label) ):
        aim_label = see_label[i]
        aim_S_f = S_f[ S_label == aim_label ]
        aim_T_f = T_f[ T_label == aim_label ]
        subfig.scatter(aim_S_f[:,0],aim_S_f[:,1],c=color[i],marker='.',linewidth=0.3,s=10)
        subfig.scatter(aim_T_f[:,0],aim_T_f[:,1],c=color[i],marker='x',linewidth=0.3,s=8)
    subfig.get_xaxis().set_visible(False)
    subfig.get_yaxis().set_visible(False)
    subfig.set_title(title,y=title_bias)


def draw_3D(subfig,title):
    base_dir = '/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/vis_data/'
    data = torch.load(base_dir+'prob_CB_cls.npy')

    import numpy as np
    data= data.detach().cpu().numpy()

    _x = np.arange(data.shape[0])
    _y = np.arange(data.shape[1])
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = data.ravel()
    bottom = np.zeros_like(top)
    width = depth = 0.50

    subfig.bar3d(x, y, bottom, width, depth, top, shade=True)
    subfig.set_title(title,y=title_bias)
    # ,x=0.65
    subfig.set_xlabel('Index of codewords')
    subfig.set_ylabel('Class labels')
    subfig.set_zlabel('Corelation')

if __name__ == '__main__':

    fig = plt.figure(figsize=(3, 3))
    save_path = './Product_Real_2.pdf'
    
    base_dir = '/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/vis_data/'
    S_f = torch.load(base_dir+'S_f.npy')
    S_label = torch.load(base_dir+'S_label.npy')
    T_f = torch.load(base_dir+'T_f.npy')
    T_label = torch.load(base_dir+'T_label.npy')
    

    # id_list = list(range(65))
    # see_label =  list(range(10,20))
    see_label =  list(range(10,14))+[25]+list(range(15,20))
    # see_label = [10,12,13,15]
    #  15 nice 
    # see_label =  list(range(50,60))
    plot_data = tsne( see_label , S_f,S_label,T_f,T_label)
    
    subfig1 = fig.add_subplot(111)
    plot_tsne(see_label, plot_data, subfig1 ,'')

    plt.tight_layout()
    plt.savefig(save_path)