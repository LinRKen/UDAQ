import sys
sys.path.append('../..')

from PIL import Image, ImageFile
import numpy as np
from tqdm import tqdm

def read_list_from_file(path):
    file = open(path,'r')
    data_list= []
    for line in file:
        img_path,label = line.split()
        data_list.append( ( img_path , int(label) ) )
    file.close()
    return data_list

def produce_pic(pic_dir,list_dir,img_size):
    data_list = read_list_from_file(list_dir)
    # for i,(img_dir,label) in tqdm(enumerate(data_list)):
    for i in tqdm( range( len(data_list) ) ):
        img_dir,label = data_list[i]
        img = Image.open( pic_dir+img_dir ).convert('RGB')

        img = img.resize( (img_size,img_size) )

        img = np.array(img)
        img = np.expand_dims(img,axis=0)

        label = np.array([label])

        if i == 0:
            data_img = img
            data_label = label
        else:
            data_img = np.concatenate((data_img, img), axis=0)
            data_label = np.concatenate((data_label, label), axis=0)

    return data_img,data_label

def get_sim(Q_label : np.array, G_label : np.array):
    Q_label = Q_label.reshape(-1,1)
    G_label = G_label.reshape(1,-1)
    Sim = Q_label==G_label
    Sim = Sim.astype(int)
    return Sim

def color_preprocessing(x_):
    
    #Normalize with mean and std of source data
    x_ = x_.astype('float32')

    x_[:, :, :, 0] = (x_[:, :, :, 0] - 125.642) / 63.01
    x_[:, :, :, 1] = (x_[:, :, :, 1] - 123.738) / 62.157
    x_[:, :, :, 2] = (x_[:, :, :, 2] - 114.46) / 66.94

    return x_


def anti_color_preprocessing(x_):
    x_ = x_.astype('float32')

    x_[:, :, :, 0] = (x_[:, :, :, 0]*63.01 + 125.642)
    x_[:, :, :, 1] = (x_[:, :, :, 1]*62.157 + 123.738) 
    x_[:, :, :, 2] = (x_[:, :, :, 2]*66.94 + 114.46)

    return x_

def get_pic_label(domain):
    file_dir = '/mnt/hdd1/zhangzhibin/dataset/DomainNet/npy_files/'
    # file_dir = '/mnt/hdd1/zhangzhibin/dataset/DomainNet/npy_files_224/'
    import os
    file_list = os.listdir(file_dir)
    aim_file = domain+'_train_pic.npy'
    if aim_file not in file_list:
        pic_dir = '/mnt/hdd1/zhangzhibin/dataset/DomainNet/'
        for name in ['train','test']:
            list_dir = '/mnt/hdd1/zhangzhibin/dataset/DomainNet/'+domain+'_'+name+'.txt'
            data_pic,data_label = produce_pic(pic_dir,list_dir,32)
            np.save(file_dir+domain+'_'+name+'_pic.npy',data_pic)
            np.save(file_dir+domain+'_'+name+'_label.npy',data_label)
    train_pic = np.load(file_dir+domain+'_train_pic.npy')
    train_label = np.load(file_dir+domain+'_train_label.npy')

    test_pic = np.load(file_dir+domain+'_test_pic.npy')
    test_label = np.load(file_dir+domain+'_test_label.npy')
    return train_pic,train_label,test_pic,test_label

        

def deal_DomainNet(S_domain,T_domain):
    print('DomainNet data load start')
    S_train_pic,S_train_label,_,_ = get_pic_label(S_domain)
    T_train_pic,_,T_test_pic,T_test_label = get_pic_label(T_domain)
    

    S_train_pic = color_preprocessing(S_train_pic)
    T_train_pic = color_preprocessing(T_train_pic)
    T_test_pic = color_preprocessing(T_test_pic)
    print('DomainNet data load OK')

    S_pic = S_train_pic
    S_label = S_train_label
    
    T_query_pic = T_test_pic
    T_query_label = T_test_label

    label_similarity = get_sim(T_query_label,S_label)
    
    return (S_pic,S_label,T_train_pic),(S_pic,T_query_pic,label_similarity)

# S_domain = 'clipart'
# T_doamin = 'clipart'
# print(S_domain,T_doamin)
# (Source_x, Source_y, Target_x),(Gallery_x, Query_x,label_similarity) = deal_DomainNet(S_domain,T_doamin)
# print(Source_x.shape)
# print(Source_y.shape)
# print(Query_x.shape)
# print(label_similarity.shape)