from cv2 import randShuffle
import torch
import numpy as np
import random
from PIL import Image, ImageFile
import os.path as osp
from datasets.domainnet import DomainNet

ImageFile.LOAD_TRUNCATED_IMAGES = True

from icecream import ic

#  torch.stack(imgs, dim=0), pids, camids, viewids, idx

class fixed_dataset(torch.utils.data.Dataset):
    def __init__(self,  f, label):
        super(fixed_dataset, self).__init__()
        self.f = torch.FloatTensor(f)
        self.label = label
        self.list = []
        for i in range(self.f.shape[0]):
            self.list.append( (self.f[i],self.label[i],i) )

    def __getitem__(self, index):
        # return torch.FloatTensor(self.f[index]), self.label[index], index
        return torch.FloatTensor(self.f[index]), self.label[index]
        #  torch.stack(imgs, dim=0), pids, camids, viewids, idx
    def get_list(self):
        return self.list

    def __len__(self):
        return self.f.shape[0]

class pair_dataset(torch.utils.data.Dataset):
    def __init__(self,data_list):
        super(pair_dataset, self).__init__()
        self.data_list = data_list

        # train_set.append(((s_img_path, t_img_path), (label,target_pseudo_label[curidx].item()), camid, trackid, (s_idx, curidx.item())))
    
    def __getitem__(self, index):
        img_pair = self.data_list[index][0]
        label_pair = self.data_list[index][1]
        idx_pair = self.data_list[index][4]
        return (img_pair[0],img_pair[1],label_pair[0],label_pair[1],idx_pair[0],idx_pair[1])

    def __len__(self):
        return len(self.data_list)

def split_DomainNet(domain_name):
    root_dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/domainnet/'
    f_dir = root_dir+domain_name+'_train/'+domain_name+'_train_feature.npy'
    label_dir = root_dir+domain_name+'_train/'+domain_name+'_train_label.npy'
    f_train = np.load(f_dir)
    label_train = np.load(label_dir)
    train_set = fixed_dataset(f_train,label_train)

    f_dir = root_dir+domain_name+'_val/'+domain_name+'_val_feature.npy'
    label_dir = root_dir+domain_name+'_val/'+domain_name+'_val_label.npy'
    f_val = np.load(f_dir)
    label_val = np.load(label_dir)
    val_set = fixed_dataset(f_val,label_val)
    return train_set,val_set,train_set

def split_OfficeHome(domain_name):
    root_dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/OfficeHome/'
    datasets = {}
    for name in ['train','query','all']:
        f_dir = root_dir+domain_name+'/'+name+'_feature.npy'
        label_dir = root_dir+domain_name+'/'+name+'_label.npy'
        features = np.load(f_dir)
        labels = np.load(label_dir)
        dataset = fixed_dataset(features,labels)
        datasets[ name ] = dataset
    return datasets['train'],datasets['query'],datasets['all']



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class img_DomainNet_dataset(torch.utils.data.Dataset):
    def __init__(self, domain_list, transform=None):
        super(img_DomainNet_dataset, self).__init__()
        self.dataset = []
        self.transform = transform
        for domain,idx_list in domain_list:
            dir = '/home/zhangzhibin/data/UDA/CDTrans-master/data/domainnet/'+domain+'.txt'
            dataset = DomainNet(root_train=dir,root_val=dir).train
            self.dataset += [ dataset[ idx ] for idx in idx_list]
    
    def __getitem__(self, index):
        img_path, pid, camid,trackid, idx = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, pid
        
    def __len__(self):
        return len(self.dataset)

def split_img_DomainNet(source_domain,target_domain,train_transform=None,val_transform=None,merge_train=False):
    dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/domainnet/'+source_domain+'/'
    S_train_idx_list = np.load(dir+'train_idx.npy')
    S_gallery_idx_list = np.load(dir+'gallery_idx.npy')

    dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/domainnet/'+target_domain+'/'
    T_train_idx_list = np.load(dir+'train_idx.npy')
    T_query_idx_list = np.load(dir+'query_idx.npy')

    if merge_train:
        trainset = img_DomainNet_dataset( [ (source_domain,S_train_idx_list), (target_domain,T_train_idx_list) ] , train_transform)
    else:
        trainset = (    img_DomainNet_dataset( [(source_domain,S_train_idx_list)]  , train_transform) , 
                        img_DomainNet_dataset( [(target_domain,T_train_idx_list)]  , train_transform) )
    query_set = img_DomainNet_dataset( [(target_domain,T_query_idx_list)]   , val_transform)
    gallery_set = img_DomainNet_dataset( [(source_domain,S_train_idx_list)] , val_transform)
    return trainset, query_set,gallery_set
    





def split_dataset(dir, reshuffle=False, tr_per_cls=0, Q_pre_cls=50):
    if reshuffle:
        domain_name = dir.split('/')[-2]
        # ic(domain_name)

        f_dir = dir+domain_name+'_feature.npy'
        label_dir = dir+domain_name+'_label.npy'
        idx_dir = dir+domain_name+'_idx.npy'

        features = np.load(f_dir)
        labels = np.load(label_dir)
        idx_list = np.load(idx_dir)

        tmp = torch.LongTensor(labels)
        cnt = torch.bincount(tmp).float()

        idx = torch.argsort(cnt)[:10]
        print(idx)
        print(cnt.max())
        print(cnt.std())
        print(cnt.mean())
        print(cnt.sum())

        exit()

        id_list = list(range(features.shape[0]))
        random.shuffle(id_list)

        n_class = np.max(labels)+1

        query_id = []
        gallery_id = []
        cnt_cls = [0]*n_class

        for id in id_list:
            f, cls = features[id], labels[id]
            if cnt_cls[cls] < Q_pre_cls:
                cnt_cls[cls] += 1
                query_id.append(id)
            else:
                gallery_id.append(id)

        train_id = gallery_id.copy()

        train_dataset = fixed_dataset(features[train_id], labels[train_id])
        query_dataset = fixed_dataset(features[query_id], labels[query_id])
        gallery_dataset = fixed_dataset(
            features[gallery_id], labels[gallery_id])

        np.save(dir+'train_features.npy', features[train_id])
        np.save(dir+'query_features.npy', features[query_id])
        np.save(dir+'gallery_features.npy', features[gallery_id])

        np.save(dir+'train_labels.npy', labels[train_id])
        np.save(dir+'query_labels.npy', labels[query_id])
        np.save(dir+'gallery_labels.npy', labels[gallery_id])

        np.save(dir+'train_idx.npy', idx_list[train_id])
        np.save(dir+'query_idx.npy', idx_list[query_id])
        np.save(dir+'gallery_idx.npy', idx_list[gallery_id])

        return train_dataset, query_dataset, gallery_dataset
    else:
        f_dir = dir+'train_features.npy'
        label_dir = dir+'train_labels.npy'
        tr_features = np.load(f_dir)
        tr_labels = np.load(label_dir)
        train_dataset = fixed_dataset(tr_features, tr_labels)

        f_dir = dir+'query_features.npy'
        label_dir = dir+'query_labels.npy'
        Q_features = np.load(f_dir)
        Q_labels = np.load(label_dir)
        query_dataset = fixed_dataset(Q_features, Q_labels)

        f_dir = dir+'gallery_features.npy'
        label_dir = dir+'gallery_labels.npy'
        G_features = np.load(f_dir)
        G_labels = np.load(label_dir)
        gallery_dataset = fixed_dataset(G_features, G_labels)
        return train_dataset, query_dataset, gallery_dataset


def pair_train_collate_fn(batch):
    b_data = zip(*batch)

    
    s_imgs,t_imgs,s_pids,t_pids,s_idx,t_idx=b_data

    img1 = torch.stack(s_imgs, dim=0)
    img2 = torch.stack(t_imgs, dim=0)
    
    s_pid = torch.tensor(s_pids, dtype=torch.long)
    t_pid = torch.tensor(t_pids, dtype=torch.long)
    
    s_idx = torch.tensor(s_idx, dtype=torch.long)
    t_idx = torch.tensor(t_idx, dtype=torch.long)
    
    return img1, img2, s_pid, t_pid, s_idx, t_idx

def get_MNIST_USPS():    
    list_dataset=[]
    # dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS/MNIST_no_crop/'
    dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS/MNIST/'
    # dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg_bn/MNIST_USPS/MNIST/'
    # dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS_mean/MNIST/'
    for name in ['all','query','all',]:
        f_dir = dir+name+'_feature.npy'
        label_dir = dir+name+'_label.npy'
        features = np.load(f_dir)
        labels = np.load(label_dir)
        dataset = fixed_dataset(features, labels)
        list_dataset.append( dataset )
    
    # dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS/USPS_no_crop/'
    dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS/USPS/'
    # dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg_bn/MNIST_USPS/USPS/'
    # dir = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS_mean/USPS/'
    for name in ['train','query','gallery',]:
        f_dir = dir+name+'_feature.npy'
        label_dir = dir+name+'_label.npy'
        features = np.load(f_dir)
        labels = np.load(label_dir)
        dataset = fixed_dataset(features, labels)
        list_dataset.append( dataset )

    return list_dataset

# a = get_MNIST_USPS()