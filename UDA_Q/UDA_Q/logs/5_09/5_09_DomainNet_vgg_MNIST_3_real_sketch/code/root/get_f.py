import sys
import os
import time
from xml.sax.handler import all_features
from cv2 import add
import numpy as np
import pickle
import glob
from datetime import datetime

from datasets import make_dataloader

# pytorch, torch vision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, dataloader
import torchvision.transforms as transforms
from torchvision import models

from icecream import ic

from config import cfg
import argparse
from tqdm import tqdm

from my_dataset import split_data_list_w_cls,write_list_to_file
from split_dataset import split_dataset

def split_list(list_dir,num_query_per_cls,n_class):
	from my_dataset import read_list_from_file,write_list_to_file
	all_list = read_list_from_file(list_dir+'all_list.txt')
	print('count of all data = ', len(all_list))
	query_list,train_list,_ = split_data_list_w_cls(num_query_per_cls,100000,all_list,n_class)
	gallery_list = train_list

	print('count of query_list = ', len(query_list))
	print('count of train_list = ', len(train_list))


	write_list_to_file( query_list,list_dir+'query_list.txt')
	write_list_to_file( train_list,list_dir+'train_list.txt')
	write_list_to_file( gallery_list,list_dir+'gallery_list.txt')

def extract_feature(data_loader,model,save_path):
	pre_time = time.time()
	cnt = 0
	with torch.no_grad():
		model.eval()
		for n_iter, (img, vid) in enumerate(tqdm(data_loader)):
			img = img.cuda()
			cnt += img.size(0)
			f = model( img )

			f = f.detach().cpu()
			vid = vid.detach().cpu()
			
			if n_iter == 0:
				features = f
				lables = vid
			else:
				features = torch.cat( (features,f))
				lables = np.concatenate((lables, vid), axis=0)
			del f
	ic(cnt)
	features = features.numpy()

	elapsed = time.time() - pre_time
	print(f"Time for get features:{elapsed//60:.0f}m{elapsed%60:.0f}s\n")

	f_addr = save_path+'_feature.npy'
	cls_addr = save_path+'_label.npy'
	
	print('write feature at '+ f_addr)
	np.save(f_addr, features)
	np.save(cls_addr, lables)


if __name__ == '__main__':

	from datasets.domainnet import DomainNet
	# from datasets.bases import ImageDataset
	import torchvision.transforms as T
	from datasets.make_dataloader import train_collate_fn

	val_transforms = T.Compose([
		# T.Resize((32, 32)),
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


	# for domain in ['painting','quickdraw','real','sketch','infograph','clipart']:
	dataset = 'DomainNet'
	if dataset == 'DomainNet':
		domain = 'quickdraw'
		print(dataset,domain)

		# dataset = DomainNet(root_train="/mnt/hdd1/zhangzhibin/dataset/DomainNet/"+domain+"_train.txt",root_val="/mnt/hdd1/zhangzhibin/dataset/DomainNet/"+domain+"_test.txt")
		from my_dataset import ImageDataset

		from base_models import model_vgg_MNIST
		model = model_vgg_MNIST()
		# model = model_vgg()
		
		modle = model.cuda()
		model.eval()

		for name in ['train','test']:
			list_dir = '/mnt/hdd1/zhangzhibin/dataset/DomainNet/'+domain+'_'+name+'.txt'
			pic_dir = '/mnt/hdd1/zhangzhibin/dataset/DomainNet/'
			dataset = ImageDataset(pic_dir,list_dir,val_transforms)
			img_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
			path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/domainnet_vgg_MNIST/'+domain+'/'
			try:
				os.mkdir(path)
			except:
				pass
			extract_feature(img_loader,model,path+name)

	elif dataset == 'OfficeHome':
		domain = 'Product'
		print(dataset,domain)
		# Art Product Real_World Clipart
		list_dir = '/mnt/hdd1/zhangzhibin/dataset/OfficeHome/image_list/'+domain+'.txt'
		from my_dataset import ImageDataset
		import os

		from base_models import model_vgg_MNIST
		model = model_vgg_MNIST()
		# model = model_vgg()
		
		modle = model.cuda()
		model.eval()

		split_dataset(list_dir)

		for name in ['all','train','test']:
			list_dir = '/mnt/hdd1/zhangzhibin/dataset/OfficeHome/image_list/'+domain+'_'+name+'.txt'
			dataset = ImageDataset('/mnt/hdd1/zhangzhibin/dataset/OfficeHome/',list_dir,val_transforms)
			img_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
			# path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/OfficeHome_resplit/'+domain+'/'
			path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/OfficeHome_vgg_MNIST/'+domain+'/'
			try:
				os.mkdir(path)
			except:
				pass
			extract_feature(img_loader,model,path+name)

		# # path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/OfficeHome/'+domain+'/'
		# path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/OfficeHome_new/'+domain+'/'
		# try:
		# 	# os.mkdir('/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/OfficeHome/'+domain)
		# 	os.mkdir('/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/OfficeHome_new/'+domain)
		# except:
		# 	pass
		
		# dataset = ImageDataset('/mnt/hdd1/zhangzhibin/dataset/OfficeHome/',list_dir,val_transforms)
		# img_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
		# extract_feature(img_loader,model,path+'all')
		# data_list = dataset.list_image
		# random.shuffle(data_list)
		
		# dataset.list_image = data_list[:500]
		# img_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
		# extract_feature(img_loader,model,path+'query')

		# dataset.list_image = data_list[500:]
		# img_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
		# extract_feature(img_loader,model,path+'train')

	elif dataset == 'MNIST_USPS':
		from my_dataset import ImageDataset

		# if True:
		if False:
			import torchvision
			from my_dataset import rewrite_pics
			normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
			val_transform = transforms.Compose([
						transforms.Resize(256),
						transforms.CenterCrop(224),
						transforms.ToTensor(),
						normalize,
					])
			root_path = '/mnt/hdd1/zhangzhibin/dataset/'
			n_pic_dir = '/mnt/hdd1/zhangzhibin/dataset/MNIST/n_pics/'
			all_list_file = open('/mnt/hdd1/zhangzhibin/dataset/MNIST/all_list.txt','w')
			train = torchvision.datasets.MNIST(root_path,train=True,download=False)
			val = torchvision.datasets.MNIST(root_path,train=False,download=False)
			cnt = rewrite_pics(train,0,n_pic_dir,all_list_file)
			cnt = rewrite_pics(val,cnt,n_pic_dir,all_list_file)

			root_path = '/mnt/hdd1/zhangzhibin/dataset/'
			n_pic_dir = '/mnt/hdd1/zhangzhibin/dataset/USPS/n_pics/'
			all_list_file = open('/mnt/hdd1/zhangzhibin/dataset/USPS/all_list.txt','w')
			train = torchvision.datasets.USPS(root_path,train=True,download=False)
			val = torchvision.datasets.USPS(root_path,train=False,download=False)
			cnt = rewrite_pics(train,0,n_pic_dir,all_list_file)
			cnt = rewrite_pics(val,cnt,n_pic_dir,all_list_file)
		
		if True:
			# list_dir = '/mnt/hdd1/zhangzhibin/dataset/MNIST/'
			# split_list(list_dir,50,10)

			# list_dir = '/mnt/hdd1/zhangzhibin/dataset/USPS/'
			# split_list(list_dir,50,10)
			split_dataset("/mnt/hdd1/zhangzhibin/dataset/MNIST/MNIST.txt")
			split_dataset('/mnt/hdd1/zhangzhibin/dataset/USPS/USPS.txt')

		from base_models import model_vgg_MNIST
		model = model_vgg_MNIST()
		# model = model_vgg()
		modle = model.cuda()
		model.eval()
		for domain in ['MNIST','USPS']:
			for name in ['all','train','test']:
				list_dir = "/mnt/hdd1/zhangzhibin/dataset/"+domain+'/'+domain+'_'+name+".txt"
				dataset = ImageDataset('/mnt/hdd1/zhangzhibin/dataset/'+domain+'/',list_dir,val_transforms)
				img_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
				# path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS/MNIST/'
				# path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS_mean/MNIST/'
				# path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS_new/MNIST/'
				path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS_resplit/'+domain+'/'
				extract_feature(img_loader,model,path+name)

		# # for name in ['all','train','query','gallery']:
		# for name in ['all','train','query','gallery']:
		# 	list_dir = "/mnt/hdd1/zhangzhibin/dataset/USPS/"+name+"_list.txt"
		# 	dataset = ImageDataset('/mnt/hdd1/zhangzhibin/dataset/USPS/',list_dir,val_transforms)
		# 	img_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
		# 	# path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS/USPS/'
		# 	# path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS_mean/USPS/'
		# 	path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS_resplit/USPS/'
		# 	extract_feature(img_loader,model,path+name)