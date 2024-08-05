import torch
import matplotlib.pyplot as plt

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
    subfig.set_title(title)
    # ,x=0.65
    subfig.set_xlabel('Index of codewords',fontsize=fontsize)
    subfig.set_ylabel('Class labels',fontsize=fontsize)
    subfig.set_zlabel('Corelation',fontsize=fontsize)


def draw_2D(subfig,title):
    base_dir = '/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/vis_data/'
    data = torch.load(base_dir+'prob_CB_cls.npy')

    data_sum = torch.sum( data, dim = 1)
    data = data[data_sum>0.5 , :]

    import numpy as np
    data= data.detach().cpu().numpy()

    data = data[:65,]

    cb_size,lable_size = data.shape

    cm = 'Reds'
    pcm = subfig.pcolormesh( data,cmap=cm)
    fig.colorbar(pcm, ax=subfig)
    plt.xticks(np.arange(0,lable_size, 1), size = 0)
    plt.yticks(np.arange(0,cb_size, 1), size =0)
    plt.grid(linewidth=0.3)
    plt.tick_params(left=False,bottom=False,labelleft=False, labelbottom=False)
    plt.xlabel('Class labels',fontsize=fontsize)
    plt.ylabel('Codewords',fontsize=fontsize)


fontsize = 16
fig = plt.figure(figsize=(7, 6))
save_path = './prob_CB_2D_part.pdf'
# subfig = fig.add_subplot(111,projection='3d')
subfig = fig.add_subplot(111)
draw_2D(subfig,'')

# plt.tight_layout()
plt.savefig(save_path)
