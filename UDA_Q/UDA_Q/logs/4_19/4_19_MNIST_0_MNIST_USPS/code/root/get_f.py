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
from base_models import model_vgg


def extract_feature(data_loader,model,save_path):
	pre_time = time.time()
	cnt = 0
	with torch.no_grad():
		model.eval()
		for n_iter, (img, vid) in enumerate(tqdm(data_loader)):
			img = img.cuda()
			cnt += img.size(0)
			f = model( img )
			
			if n_iter == 0:
				features = f
				lables = vid
			else:
				features = torch.cat( (features,f))
				lables = np.concatenate((lables, vid), axis=0)
	ic(cnt)
	features = features.detach().cpu().numpy()

	elapsed = time.time() - pre_time
	print(f"Time for get features:{elapsed//60:.0f}m{elapsed%60:.0f}s\n")

	f_addr = save_path+'_feature.npy'
	cls_addr = save_path+'_label.npy'
	
	print('write feature at '+ f_addr)
	np.save(f_addr, features)
	np.save(cls_addr, lables)


if __name__ == '__main__':	
	model = model_vgg()
	model.eval()
	model.cuda()

	from datasets.domainnet import DomainNet
	# from datasets.bases import ImageDataset
	import torchvision.transforms as T
	from datasets.make_dataloader import train_collate_fn

	val_transforms = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


	# for domain in ['painting','quickdraw','real','sketch','infograph','clipart']:
	dataset = 'MNIST_USPS'
	if dataset == 'DomainNet':
		domain = 'clipart'
		print(domain)

		dataset = DomainNet(root_train="/mnt/hdd1/zhangzhibin/dataset/DomainNet/"+domain+"_train.txt",root_val="/mnt/hdd1/zhangzhibin/dataset/DomainNet/"+domain+"_test.txt")
		train_datset = ImageDataset(dataset.train, val_transforms)
		val_datset = ImageDataset(dataset.test, val_transforms)
		img_loader = DataLoader(
			train_datset, batch_size=64, shuffle=False, num_workers=4,
			collate_fn=train_collate_fn
		)
		extract_feature( img_loader , model , domain+'_train')

		img_loader = DataLoader(
			val_datset, batch_size=64, shuffle=False, num_workers=4,
			collate_fn=train_collate_fn
		)
		extract_feature( img_loader , model , domain+'_val')
	elif dataset == 'OfficeHome':
		domain = 'Real_World'
		# Art Product Real_World Clipart
		list_dir = '/mnt/hdd1/zhangzhibin/dataset/OfficeHome/image_list/'+domain+'.txt'
		from my_dataset import read_list_from_file,ImageDataset
		from PIL import Image
		import random 
		import os
		random.seed(0)

		path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/OfficeHome/'+domain+'/'
		try:
			os.mkdir('/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/OfficeHome/'+domain)
		except:
			pass
		
		dataset = ImageDataset('/mnt/hdd1/zhangzhibin/dataset/OfficeHome/',list_dir,val_transforms)
		img_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
		extract_feature(img_loader,model,path+'all')
		data_list = dataset.list_image
		random.shuffle(data_list)
	elif dataset == 'MNIST_USPS':
		from my_dataset import ImageDataset

		if False:
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

			list_dir = '/mnt/hdd1/zhangzhibin/dataset/MNIST/'
			split_list(list_dir,500)

			list_dir = '/mnt/hdd1/zhangzhibin/dataset/USPS/'
			split_list(list_dir,500)

		for name in ['all','train','query','gallery']:
			list_dir = "/mnt/hdd1/zhangzhibin/dataset/MNIST/"+name+"_list.txt"
			dataset = ImageDataset('/mnt/hdd1/zhangzhibin/dataset/MNIST/',list_dir,val_transforms)
			img_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
			path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS/MNIST/'
			extract_feature(img_loader,model,path+name)

		for name in ['all','train','query','gallery']:
			list_dir = "/mnt/hdd1/zhangzhibin/dataset/USPS/"+name+"_list.txt"
			dataset = ImageDataset('/mnt/hdd1/zhangzhibin/dataset/USPS/',list_dir,val_transforms)
			img_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
			path = '/home/zhangzhibin/data/UDA/CDTrans-master/fixed_f/vgg/MNIST_USPS/USPS/'
			extract_feature(img_loader,model,path+name)