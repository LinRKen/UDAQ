
from csv import list_dialects
from cv2 import randShuffle
from sqlalchemy import all_
import torch
import numpy as np
import random
from PIL import Image, ImageFile
import os.path as osp
from datasets.domainnet import DomainNet

ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
from icecream import ic


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,root_path,  list_path  , transform=None ): 
        super(ImageDataset,self).__init__()
        self.root_path = root_path 
        # '/mnt/hdd1/zhangzhibin/dataset/CIFAR-10/'
        self.list_image  = read_list_from_file(list_path)
        self.transform = transform

    def __getitem__(self, index):    

        fn, label = self.list_image[index] 
        
        label = int( label )
        fn = self.root_path+fn
        img = Image.open( fn ).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return img , label

    def __len__(self): 
        return len( self.list_image )


def rewrite_pics(dataset,st_id,n_pic_dir,list_file):
    cnt = st_id
    for id in tqdm( range( dataset.__len__() )):
        pic,label = dataset[id]

        pic_adr = n_pic_dir+str(cnt)+'.jpg'
        pic.save(pic_adr,quality=95,subsampling=0)

        pic_adr = 'n_pics/'+str(cnt)+'.jpg'
        list_file.write( pic_adr+' '+str(label)+'\n')
        cnt+=1
    return cnt

def split_data_list(query_cnt,train_cnt,gallery_cnt,all_list):
    import random
    random.seed(0)
    random.shuffle(all_list)

    query_list = []
    train_list = []
    gallery_list = []
    cnt_cls = [0]*10000

    for i in range( len(all_list) ):
        label = int(all_list[i][1])
        if cnt_cls[label] < query_cnt:
            query_list.append( all_list[i] )
        elif cnt_cls[label] < query_cnt+train_cnt:
            train_list.append( all_list[i] )
        else:
            gallery_list.append( all_list[i] )
        cnt_cls[label] += 1
    return query_list,train_list,gallery_list

def write_list_to_file(data_list,path):
    file = open(path,'w')
    for (img_path,label) in data_list:
        file.write( img_path+' '+str(label)+'\n')
    file.close()

def read_list_from_file(path):
    file = open(path,'r')
    data_list= []
    for line in file:
        img_path,label = line.split()
        data_list.append( ( img_path , int(label) ) )
    file.close()
    return data_list

def get_CIFAR10():
    list_dir = '/mnt/hdd1/zhangzhibin/dataset/CIFAR-10/'
    import os
    list_files = os.listdir(list_dir)
    if 'all_label.txt' not in list_files:
        import torchvision
        train = torchvision.datasets.CIFAR10(root='/mnt/hdd1/zhangzhibin/dataset/CIFAR-10/', train= True,download = False)
        val = torchvision.datasets.CIFAR10(root='/mnt/hdd1/zhangzhibin/dataset/CIFAR-10/', train= False,download = False)
        
        all_list_file =open(list_dir+'all_label.txt','w')
        n_pic_dir = '/mnt/hdd1/zhangzhibin/dataset/CIFAR-10/n_pics/'

        cnt = rewrite_pics(train,0,n_pic_dir,all_list_file)
        cnt = rewrite_pics(val,cnt,n_pic_dir,all_list_file)
        print('count of all data = ',cnt)
        all_list_file.close()

    if 'train_list.txt' not in list_files or True:
        all_list = read_list_from_file(list_dir+'all_label.txt')
        print('count of all data = ', len(all_list))

        query_list,train_list,gallery_list = split_data_list(100,500,5400,all_list)
        write_list_to_file( query_list,list_dir+'query_list.txt')
        write_list_to_file( train_list,list_dir+'train_list.txt')
        write_list_to_file( gallery_list,list_dir+'gallery_list.txt')
    
    from torchvision import transforms
    import torchvision

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.3, contrast=0, saturation=0.1, hue=0),
            transforms.RandomResizedCrop( 224 ,scale=(0.7,1.3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    root_path = '/mnt/hdd1/zhangzhibin/dataset/CIFAR-10/'
    query_set = ImageDataset(root_path,list_dir+'query_list.txt',val_transform)
    train_set = ImageDataset(root_path,list_dir+'train_list.txt',transform)
    gallery_set = ImageDataset(root_path,list_dir+'gallery_list.txt',val_transform)
    return query_set,train_set,gallery_set